class PromptBuilder:

    def __init__(self, npc):
        self.npc = npc

    def build(self, query, memories, response_type):

        personality = self.npc.personality

        prompt = f"""
You are {self.npc.name}, a {self.npc.role}.

Personality:
- Honesty: {personality.honesty}
- Courage: {personality.courage}
- Intelligence: {personality.intelligence}
- Kindness: {personality.kindness}
- Suspicion: {personality.suspicion}
- Confidence: {personality.confidence}

Goals:
{self.npc.goals}

Secrets:
{self.npc.secrets}

Relevant memories:
"""

        if memories:
            for memory in memories:
                prompt += f"- {memory.content}\n"
        else:
            prompt += "- No relevant memories.\n"

        prompt += f"""
Player question:
{query}

Response policy:
{response_type}

Instructions:
- Stay in character.
- Use the provided memories as the basis for your response.
- Do not invent facts.
- Follow the response policy.
- Respond naturally and concisely.
"""

        return prompt


if __name__ == "__main__":
    print("PromptBuilder loaded successfully.")