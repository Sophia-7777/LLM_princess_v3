import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "llm/lora/output_v3"


NPCS = {
    "Emma": {
        "role": "maid",
        "style": (
            "extremely polite, timid, apologetic, indirect, "
            "and conflict-avoidant"
        ),
    },
    "Edward": {
        "role": "guard",
        "style": (
            "rude, blunt, impatient, dismissive, "
            "and aggressive"
        ),
    },
    "Lily": {
        "role": "servant",
        "style": (
            "extremely energetic, emotional, talkative, "
            "spontaneous, and easily distracted"
        ),
    },
    "Barbie": {
        "role": "noblewoman",
        "style": (
            "arrogant, aristocratic, condescending, "
            "elegant, and socially superior"
        ),
    },
}


# QUESTIONS = [
#     "Where were you last night?",
#     "Did you see the princess?",
#     "You are hiding something.",
#     "Tell me the truth.",
# ]
QUESTIONS = [
    "What do you think about the royal family?",
    "Why were you in the garden this morning?",
    "What is your favorite place in the palace?",
    "Who do you trust?",
    "What do you usually do during the evening?",
    "What do you think of the king?",
    "Why do people like the princess?",
    "What would you do if someone accused you?",
]

# def build_prompt(npc_name, npc_info, question):
#     return f"""
# You are {npc_name}, a {npc_info["role"]}.

# Your speaking style is:
# {npc_info["style"]}

# Stay completely in character.

# Player question:
# {question}

# Respond naturally and concisely.
# Do not explain your personality.
# """

def build_prompt(npc_name, npc_info, question):
    return f"""
You are {npc_name}, a {npc_info["role"]}.

Stay in character.

Player question:
{question}

Respond naturally and concisely.
"""


print("=" * 60)
print("Loading tokenizer...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("=" * 60)
print("Loading base model...")
print("=" * 60)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

base_model.eval()


print("=" * 60)
print("Loading LoRA adapter...")
print("=" * 60)

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
)

lora_model.eval()

print("LoRA model loaded.")


def generate(model, prompt, max_new_tokens=80):

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

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
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )

    input_length = inputs["input_ids"].shape[1]

    output_ids = outputs[0][input_length:]

    response = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    )

    return response.strip()


for question in QUESTIONS:

    print("\n")
    print("=" * 60)
    print("QUESTION:")
    print(question)
    print("=" * 60)

    for npc_name, npc_info in NPCS.items():

        prompt = build_prompt(
            npc_name,
            npc_info,
            question,
        )

        base_response = generate(
            base_model,
            prompt,
        )

        lora_response = generate(
            lora_model,
            prompt,
        )

        print("\n" + "-" * 60)
        print(f"NPC: {npc_name}")

        print("\n[BASE]")
        print(base_response)

        print("\n[LORA]")
        print(lora_response)