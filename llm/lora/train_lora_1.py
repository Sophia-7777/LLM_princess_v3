import json
import os

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_PATH = "llm/lora/data/train.json"

OUTPUT_DIR = "llm/lora/output"

MAX_LENGTH = 512


# ============================================================
# 2. Load training data
# ============================================================

print("=" * 60)
print("Loading training data...")
print("=" * 60)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Number of training samples: {len(data)}")

npc_names = sorted(set(item["npc"] for item in data))
print(f"NPCs: {npc_names}")


# ============================================================
# 3. Convert data into training text
# ============================================================

def build_training_text(item):
    """
    Convert one NPC training example into a Qwen
    instruction-style conversation.

    The NPC identity and style are explicitly included
    so that one Shared LoRA can learn multiple styles.
    """

    messages = [
        {
            "role": "system",
            "content": (
                f"You are {item['npc']}, a {item['role']}.\n"
                f"Your speaking style is: {item['style']}\n\n"
                "Stay in character. Respond according to your "
                "speaking style."
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
# 4. Load tokenizer
# ============================================================

print("=" * 60)
print("Loading tokenizer...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")


# ============================================================
# 5. Apply Qwen chat template
# ============================================================

training_texts = []

for item in data:

    messages = build_training_text(item)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    training_texts.append(text)


dataset = Dataset.from_dict(
    {
        "text": training_texts
    }
)

print(f"Dataset created: {len(dataset)} samples")


# ============================================================
# 6. Tokenization
# ============================================================

def tokenize_function(examples):

    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    return result


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

print("Tokenization completed.")


# ============================================================
# 7. Load base model
# ============================================================

print("=" * 60)
print("Loading base model...")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

model.config.use_cache = False

print("Base model loaded.")


# ============================================================
# 8. Configure LoRA
# ============================================================

print("=" * 60)
print("Configuring LoRA...")
print("=" * 60)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,

    # Size of LoRA update matrices
    r=8,

    # Scaling factor
    lora_alpha=16,

    # Regularization
    lora_dropout=0.05,

    # Qwen attention projection modules
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

print("LoRA configured.")

model.print_trainable_parameters()


# ============================================================
# 9. Training arguments
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # CPU training
    use_cpu=True,

    # Small dataset
    num_train_epochs=5,

    # CPU memory
    per_device_train_batch_size=1,

    # Simulate larger batch
    gradient_accumulation_steps=4,

    # Learning rate for LoRA
    learning_rate=2e-4,

    # Weight decay
    weight_decay=0.01,

    # Logging
    logging_steps=1,

    # Save checkpoint
    save_strategy="epoch",

    # Keep only the latest checkpoint
    save_total_limit=2,

    # Do not evaluate yet
    eval_strategy="no",

    # Do not use fp16/bf16 on CPU
    fp16=False,
    bf16=False,

    # Useful for small dataset
    report_to="none",

    # Keep training deterministic
    seed=42,
)


# ============================================================
# 10. Data collator
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ============================================================
# 11. Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)


# ============================================================
# 12. Start training
# ============================================================

print("=" * 60)
print("Starting LoRA training...")
print("=" * 60)

trainer.train()


# ============================================================
# 13. Save LoRA adapter
# ============================================================

print("=" * 60)
print("Saving LoRA adapter...")
print("=" * 60)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA adapter saved.")
print(f"Output directory: {OUTPUT_DIR}")


# ============================================================
# 14. Final information
# ============================================================

print("=" * 60)
print("Training completed.")
print("=" * 60)

print(f"Model: {MODEL_NAME}")
print(f"Training samples: {len(data)}")
print(f"NPCs: {', '.join(npc_names)}")
print(f"Output: {OUTPUT_DIR}")