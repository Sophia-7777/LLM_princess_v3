import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMClient:

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name

        print(f"Loading model: {model_name}")
        # 字符串变为token id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

        self.model.eval()

        print("Model loaded.")

    def generate(self, prompt, max_new_tokens=128):

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        # 转格式到符合model的接口
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # print("text:",text)

        # 转为向量
        model_inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )
        # print("model_inputs:", model_inputs)

        with torch.no_grad(): # 不记录梯度

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True, # 随机采样生成
                temperature=0.7, # 控制随机性
                top_p=0.8 # 从累计概率达到约 80% 的候选 Token 中进行采样
            )

        input_length = model_inputs["input_ids"].shape[1]

        output_ids = generated_ids[
            0
        ][
            input_length:
        ]

        response = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        )

        return response.strip()


if __name__ == "__main__":

    client = LLMClient()

    result = client.generate(
        "Introduce yourself in one sentence."
    )

    print("\nModel response:")
    print(result)