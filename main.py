from retriever import Retriever
from policy import Policy
from llm.prompts import PromptBuilder
from llm.client import LLMClient
from npc import NPC
from personality import Personality
from memory import Memory
from embedding import Embedding

class Game:

    def __init__(self, npc, memories):
        self.npc = npc
        self.memories = memories

        self.retriever = Retriever(memories, embedding_model=Embedding())
        self.policy = Policy(npc, embedding_model=Embedding())
        self.prompt_builder = PromptBuilder(npc)
        self.llm_client = LLMClient()

    def ask_npc(self, query):
        # 1. Retrieve relevant memories
        relevant_memories = self.retriever.retrieve(query, top_k=5)
        print("\nRetrieved memories:")
        for memory, score in relevant_memories:
            print(
                f"Score: {score:.4f} | "
                f"{memory.content}"
            )
        # 2. Decide how the NPC should respond
        response_type = self.policy.decide(
            query,
            relevant_memories
        )
        print("response_type:", response_type)
        # 3. Build the prompt
        prompt = self.prompt_builder.build(
            query,
            memories,
            response_type
        )
        print("prompt:", prompt)
        print("qqqqqq")
        # 4. Generate the response
        response = self.llm_client.generate(prompt)

        return response


if __name__ == "__main__":
    personality = Personality(
        honesty=0.5,
        courage=0.5,
        intelligence=0.7,
        kindness=0.6,
        suspicion=0.5,
        confidence=0.5
    )

    npc = NPC(
        npc_id="npc001",
        name="Emma",
        role="Maid",
        personality=personality
    )

    npc.add_secret(
        # "I secretly took Barbie's letter."
        "I saw Barbie walking toward the rose garden."
    )

    memory1 = Memory(
        id="m1",
        npc_id="npc001",
        # content="I saw Barbie walking toward the rose garden.",
        content="I took Barbie's letter.",
        event_id=None,
        timestamp=100,
        importance=0.8
    )

    memory2 = Memory(
        id="m2",
        npc_id="npc001",
        content="Barbie dropped a letter near the rose garden.",
        event_id=None,
        timestamp=110,
        importance=0.9
    )

    memory3 = Memory(
        id="m3",
        npc_id="npc001",
        content="I picked up something from the ground.",
        event_id=None,
        timestamp=115,
        importance=0.7
    )

    memories = [
        memory1,
        memory2,
        memory3
    ]

    game = Game(
        npc=npc,
        memories=memories
    )
    ask = "Where did you see Barbie?"

    print("\nPlayer:")
    print(ask)

    result = game.ask_npc(ask)

    print("\nEmma:")
    print(result)

    print("\nGame loaded successfully.")