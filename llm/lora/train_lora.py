import json
import os

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_PATH = "llm/lora/data/train_v3.json"
OUTPUT_DIR = "llm/lora/output_v4"

MAX_LENGTH = 512

NUM_EPOCHS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4

SEED = 42


# ============================================================
# 2. Load training data
# ============================================================

print("=" * 70)
print("Loading training data...")
print("=" * 70)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Number of training examples: {len(data)}")


# ============================================================
# 3. Basic dataset validation
# ============================================================

required_fields = {"npc", "user", "assistant"}

for i, item in enumerate(data):

    if not required_fields.issubset(item.keys()):
        raise ValueError(
            f"Example {i} is missing required fields. "
            f"Expected: {required_fields}"
        )

    if not item["npc"].strip():
        raise ValueError(f"Example {i}: npc is empty.")

    if not item["user"].strip():
        raise ValueError(f"Example {i}: user is empty.")

    if not item["assistant"].strip():
        raise ValueError(f"Example {i}: assistant is empty.")


print("Dataset validation passed.")


# ============================================================
# 4. Show NPC distribution
# ============================================================

npc_counts = {}

for item in data:
    npc = item["npc"]
    npc_counts[npc] = npc_counts.get(npc, 0) + 1

print("\nNPC distribution:")

for npc, count in npc_counts.items():
    print(f"  {npc}: {count}")


# ============================================================
# 5. Load tokenizer
# ============================================================

print("\n" + "=" * 70)
print("Loading tokenizer...")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")


# ============================================================
# 6. Build chat messages
# ============================================================

def build_messages(item):

    messages = [
        {
            "role": "system",
            "content": (
                f"You are {item['npc']}.\n\n"
                "Goals:\n"
                "- No specific goals provided.\n\n"
                "Secrets:\n"
                "- No known secrets provided.\n\n"
                "Relevant memories:\n"
                "- No relevant memories.\n\n"
                "Response policy:\n"
                "Answer naturally and stay in character.\n"
            ),
        },
        {
            "role": "user",
            "content": item["user"],
        },
        {
            "role": "assistant",
            "content": item["assistant"],
        },
    ]

    return messages


# ============================================================
# 7. Tokenization
#
# Important:
# We calculate loss ONLY on assistant tokens.
# ============================================================

def tokenize_example(item):

    messages = build_messages(item)

    # --------------------------------------------------------
    # Full conversation:
    #
    # system
    # user
    # assistant
    # --------------------------------------------------------

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # --------------------------------------------------------
    # Prompt only:
    #
    # system
    # user
    #
    # This is used to determine where assistant generation
    # starts.
    # --------------------------------------------------------

    prompt_messages = messages[:2]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # --------------------------------------------------------
    # Tokenize full conversation
    # --------------------------------------------------------

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    # --------------------------------------------------------
    # Tokenize prompt
    # --------------------------------------------------------

    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    prompt_length = len(prompt_tokens["input_ids"])

    # --------------------------------------------------------
    # Create labels
    #
    # -100 means:
    # "Do not calculate loss for this token."
    #
    # Therefore:
    #
    # system + user -> -100
    # assistant      -> actual token IDs
    # --------------------------------------------------------

    labels = [-100] * len(input_ids)

    for i in range(prompt_length, len(input_ids)):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# 8. Convert dataset
# ============================================================

print("\n" + "=" * 70)
print("Tokenizing dataset...")
print("=" * 70)

dataset = Dataset.from_list(data)

tokenized_dataset = dataset.map(
    tokenize_example,
    remove_columns=dataset.column_names,
)

print("Tokenization completed.")


# ============================================================
# 9. Verify labels
# ============================================================

print("\n" + "=" * 70)
print("Checking assistant-only labels...")
print("=" * 70)

for i in range(min(5, len(tokenized_dataset))):

    labels = tokenized_dataset[i]["labels"]

    valid_labels = [
        token for token in labels
        if token != -100
    ]

    if len(valid_labels) == 0:
        raise ValueError(
            f"Example {i} has no assistant labels!"
        )

    print(
        f"Example {i}: "
        f"{len(valid_labels)} assistant tokens"
    )


# ============================================================
# 10. Load Base Model
# ============================================================

print("\n" + "=" * 70)
print("Loading base model...")
print("=" * 70)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

model.config.use_cache = False

print("Base model loaded.")


# ============================================================
# 11. Configure LoRA
# ============================================================

print("\n" + "=" * 70)
print("Configuring LoRA...")
print("=" * 70)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,

    # LoRA rank
    r=8,

    # LoRA scaling
    lora_alpha=16,

    # Dropout
    lora_dropout=0.05,

    # Qwen attention projection layers
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    bias="none",
)


model = get_peft_model(
    model,
    lora_config,
)


# ============================================================
# 12. Print trainable parameters
# ============================================================

print("\nTrainable parameters:")

model.print_trainable_parameters()


# ============================================================
# 13. Custom data collator
# ============================================================

class CausalLMDataCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):

        max_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        pad_token_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:

            current_length = len(
                feature["input_ids"]
            )

            padding_length = (
                max_length - current_length
            )

            input_ids.append(
                feature["input_ids"]
                + [pad_token_id] * padding_length
            )

            attention_mask.append(
                feature["attention_mask"]
                + [0] * padding_length
            )

            labels.append(
                feature["labels"]
                + [-100] * padding_length
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),

            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),

            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


data_collator = CausalLMDataCollator(tokenizer)


# ============================================================
# 14. Training arguments
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    use_cpu=True,

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    learning_rate=LEARNING_RATE,

    weight_decay=0.01,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps=1,

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_strategy="epoch",

    save_total_limit=2,

    # --------------------------------------------------------
    # No evaluation dataset for now
    # --------------------------------------------------------

    eval_strategy="no",

    # --------------------------------------------------------
    # CPU -> no FP16/BF16
    # --------------------------------------------------------

    fp16=False,

    bf16=False,

    # --------------------------------------------------------
    # Disable external logging
    # --------------------------------------------------------

    report_to="none",

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=SEED,
)


# ============================================================
# 15. Trainer
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset,

    data_collator=data_collator,
)


# ============================================================
# 16. Start training
# ============================================================

print("\n" + "=" * 70)
print("Starting LoRA v2 training...")
print("=" * 70)

trainer.train()


# ============================================================
# 17. Save LoRA adapter
# ============================================================

print("\n" + "=" * 70)
print("Saving LoRA v2...")
print("=" * 70)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)


# ============================================================
# 18. Finish
# ============================================================

print("\n" + "=" * 70)
print("Training completed.")
print("=" * 70)

print(f"LoRA adapter saved to:")
print(f"  {OUTPUT_DIR}")

print("\nBase model was NOT modified.")
print("Only LoRA adapter parameters were trained.")