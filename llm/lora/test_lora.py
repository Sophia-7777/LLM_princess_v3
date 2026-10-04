import json
import gc
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "llm/lora/output_v4"
DATA_PATH = "llm/lora/data/train_v3.json"

# =========================================================
# Settings
# =========================================================

# 原始训练数据最多测试多少条
MAX_TEST_SAMPLES = 4

# 新问题
# 每一个问题都会同时询问四个 NPC
NEW_QUESTIONS = [
    "What would you do if someone betrayed you?",
    "How do you react when someone criticizes you?",
    "What makes you trust another person?",
    "What would you do if you made a serious mistake?",
    "How would you react if someone insulted you?",
]

NPCS = [
    "Sophia",
    "Barney",
    "Lucas",
    "Jack",
]


# =========================================================
# Load original data
# =========================================================

print("=" * 60)
print("Loading test data...")
print("=" * 60)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


# 只取前 MAX_TEST_SAMPLES 条
data = data[:MAX_TEST_SAMPLES]

print(f"Loaded {len(data)} original test samples.")


# =========================================================
# Build original test questions
# =========================================================

QUESTIONS = []

for item in data:

    QUESTIONS.append({
        "npc": item["npc"],
        "question": item["user"],
        "reference": item["assistant"],
        "type": "original",
    })


# =========================================================
# Add new questions
# =========================================================

for question in NEW_QUESTIONS:

    for npc in NPCS:

        QUESTIONS.append({
            "npc": npc,
            "question": question,
            "reference": None,
            "type": "new",
        })


print(
    f"Total test cases: {len(QUESTIONS)}"
)

print(
    f"Original samples: {len(data)}"
)

print(
    f"New questions: "
    f"{len(NEW_QUESTIONS)} questions × "
    f"{len(NPCS)} NPCs = "
    f"{len(NEW_QUESTIONS) * len(NPCS)} cases"
)


# =========================================================
# Build prompt
# =========================================================

def build_prompt(npc_name, question):

    messages = [
        {
            "role": "system",
            "content": (
                f"You are {npc_name}.\n\n"
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
            "content": question,
        },
    ]

    return messages

def build_messages(item):

    

    return messages
# =========================================================
# Load tokenizer
# =========================================================

print("\n")
print("=" * 60)
print("Loading tokenizer...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================================================
# Generate
# =========================================================

def generate(model, prompt, max_new_tokens=80):

    text = tokenizer.apply_chat_template(
        prompt,
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

            # 保持和之前测试一致
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
# Phase 1
# PURE BASE MODEL
# =========================================================

print("\n")
print("=" * 60)
print("PHASE 1: Loading PURE BASE MODEL")
print("=" * 60)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

base_model.eval()

print("Pure Base model loaded.")
print("NO LoRA adapter is attached.")


# =========================================================
# Generate Base results
# =========================================================

base_results = []

for i, item in enumerate(QUESTIONS):

    npc_name = item["npc"]
    question = item["question"]

    prompt = build_prompt(
        npc_name,
        question,
    )

    print(
        f"\rGenerating BASE: "
        f"{i + 1}/{len(QUESTIONS)}",
        end="",
        flush=True,
    )

    response = generate(
        base_model,
        prompt,
    )

    base_results.append(response)


print("\nBASE generation completed.")


# =========================================================
# Unload PURE BASE
# =========================================================

del base_model

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nPure Base model unloaded.")


# =========================================================
# Phase 2
# BASE + LoRA
# =========================================================

print("\n")
print("=" * 60)
print("PHASE 2: Loading BASE + LoRA")
print("=" * 60)

base_for_lora = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

base_for_lora.eval()


lora_model = PeftModel.from_pretrained(
    base_for_lora,
    LORA_PATH,
)

lora_model.eval()

print("LoRA model loaded.")
print(f"LoRA adapter: {LORA_PATH}")


# =========================================================
# Generate LoRA results
# =========================================================

lora_results = []

for i, item in enumerate(QUESTIONS):

    npc_name = item["npc"]
    question = item["question"]

    prompt = build_prompt(
        npc_name,
        question,
    )

    print(
        f"\rGenerating LORA: "
        f"{i + 1}/{len(QUESTIONS)}",
        end="",
        flush=True,
    )

    response = generate(
        lora_model,
        prompt,
    )

    lora_results.append(response)


print("\nLORA generation completed.")


# =========================================================
# Final comparison
# =========================================================

print("\n")
print("=" * 70)
print("BASE vs LORA RESULTS")
print("=" * 70)


for i, item in enumerate(QUESTIONS):

    npc_name = item["npc"]
    question = item["question"]
    reference_answer = item["reference"]
    test_type = item["type"]

    print("\n")
    print("=" * 70)

    print(
        f"TEST SAMPLE: {i + 1}/{len(QUESTIONS)}"
    )

    print(
        f"TYPE: {test_type.upper()}"
    )

    print("\nNPC:")
    print(npc_name)

    print("\nQUESTION:")
    print(question)

    print("=" * 70)

    # -----------------------------------------------------
    # Reference
    # -----------------------------------------------------

    print("\n[REFERENCE]")

    if reference_answer is not None:
        print(reference_answer)
    else:
        print("N/A - New test question")

    # -----------------------------------------------------
    # Pure Base
    # -----------------------------------------------------

    print("\n[PURE BASE]")
    print(base_results[i])

    # -----------------------------------------------------
    # LoRA
    # -----------------------------------------------------

    print("\n[LORA]")
    print(lora_results[i])

    print("=" * 70)