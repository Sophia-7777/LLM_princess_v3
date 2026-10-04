class PromptBuilder:

    def __init__(self, npc):
        self.npc = npc

    def build(self, query, memories, response_type):

        personality = self.npc.personality

        system_prompt = f"""
        You are {self.npc.name}.

        Goals:
        {self.npc.goals}
        
        Secrets:
        {self.npc.secrets}

        Relevant memories:
        """
        if memories:
            for memory in memories:
                system_prompt += f"- {memory.content}\n"
        else:
            system_prompt += "- No relevant memories.\n"

        system_prompt += f"""
        Response policy:
        {response_type}
        \n
        Stay in character.
        """

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]

        return messages


if __name__ == "__main__":
    print("PromptBuilder loaded successfully.")