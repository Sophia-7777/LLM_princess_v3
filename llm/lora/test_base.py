import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "llm/lora/data/train_v3.json"


# =========================================================
# Load test data
# =========================================================

print("=" * 60)
print("Loading data...")
print("=" * 60)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples.")


# =========================================================
# Load tokenizer
# =========================================================

print("=" * 60)
print("Loading tokenizer...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================================================
# Load BASE model only
# =========================================================

print("=" * 60)
print("Loading Qwen Base model...")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

model.eval()

print("Base model loaded.")


# =========================================================
# Build prompt
# =========================================================

def build_prompt(npc_name, question):

    messages = [
        {
            "role": "system",
            "content": f"You are {npc_name}.\nStay in character."
        },
        {
            "role": "user",
            "content": question
        }
    ]

    return messages


# =========================================================
# Generate response
# =========================================================

def generate(npc_name, question, max_new_tokens=80):

    # -----------------------------------------------------
    # IMPORTANT:
    # This is EXACTLY the same prompt structure
    # as your previous test code.
    # -----------------------------------------------------

    messages = build_prompt(
        npc_name,
        question,
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,

            # Same generation settings as your original code
            do_sample=True,
            temperature=0.7,
            top_p=0.8,

            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]

    output_ids = outputs[0][input_length:]

    response = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    )

    return response.strip()


# =========================================================
# Run
# =========================================================

for item in data:

    npc_name = item["npc"]
    question = item["user"]

    response = generate(
        npc_name,
        question,
    )

    print("\n" + "=" * 60)
    print("NPC:", npc_name)

    print("\nQUESTION:")
    print(question)

    print("\n[QWEN BASE]")
    print(response)