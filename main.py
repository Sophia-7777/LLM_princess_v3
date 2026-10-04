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
        retrieved_memory_objects = [
            memory for memory, score in relevant_memories
        ]
        # print("retrieved_memory_objects: ", retrieved_memory_objects)
        messages = self.prompt_builder.build(
            query,
            retrieved_memory_objects,
            response_type
        )
        # print("messages:", messages)
        print("qqqqqq")
        # 4. Generate the response
        response = self.llm_client.generate(messages)

        return response


if __name__ == "__main__":

    # ==============================
    # Sophia
    # ==============================

    sophia_personality = Personality(
        honesty=0.8,
        courage=0.7,
        intelligence=0.9,
        kindness=0.5,
        suspicion=0.8,
        confidence=0.8
    )

    sophia = NPC(
        npc_id="sophia_001",
        name="Sophia",
        role="refined young woman",
        personality=sophia_personality
    )

    sophia.add_goal("Maintain control of the situation")
    sophia.add_goal("Protect her reputation")

    sophia.add_secret("Sophia does not easily trust strangers.")

    sophia_memories = [
        Memory(
            id="sophia_memory_001",
            npc_id="sophia_001",
            content="The player arrived in the village yesterday.",
            timestamp=100,
            importance=0.7,
            event_id="sophia_event_001"
        ),
        Memory(
            id="sophia_memory_002",
            npc_id="sophia_001",
            content="Sophia believes the player may be hiding something.",
            timestamp=200,
            importance=0.9,
            event_id="sophia_event_002"
        ),
        Memory(
            id="sophia_memory_003",
            npc_id="sophia_001",
            content="Sophia promised to investigate the player's intentions.",
            timestamp=300,
            importance=0.8,
            event_id="sophia_event_003"
        )
    ]

    # ==============================
    # Barney
    # ==============================

    barney_personality = Personality(
        honesty=0.8,
        courage=0.9,
        intelligence=0.5,
        kindness=0.2,
        suspicion=0.7,
        confidence=0.9
    )

    barney = NPC(
        npc_id="barney_001",
        name="Barney",
        role="rough village man",
        personality=barney_personality
    )

    barney.add_goal("Get what he wants quickly")
    barney.add_goal("Avoid looking weak")

    barney.add_secret("Barney is afraid of being humiliated.")

    barney_memories = [
        Memory(
            id="barney_memory_001",
            npc_id="barney_001",
            content="The player interrupted Barney while he was working.",
            timestamp=100,
            importance=0.6,
            event_id="barney_event_001"
        ),
        Memory(
            id="barney_memory_002",
            npc_id="barney_001",
            content="Barney does not trust the player's intentions.",
            timestamp=200,
            importance=0.8,
            event_id="barney_event_002"
        ),
        Memory(
            id="barney_memory_003",
            npc_id="barney_001",
            content="Barney warned the player not to cause trouble in the village.",
            timestamp=300,
            importance=0.9,
            event_id="barney_event_003"
        )
    ]

    # ==============================
    # Lucas
    # ==============================

    lucas_personality = Personality(
        honesty=0.7,
        courage=0.3,
        intelligence=0.8,
        kindness=0.6,
        suspicion=0.7,
        confidence=0.3
    )

    lucas = NPC(
        npc_id="lucas_001",
        name="Lucas",
        role="timid young man",
        personality=lucas_personality
    )

    lucas.add_goal("Avoid unnecessary conflict")
    lucas.add_goal("Protect his secrets")

    lucas.add_secret("Lucas is afraid of being abandoned.")
    lucas.add_secret("Lucas knows something important about the village.")

    lucas_memories = [
        Memory(
            id="lucas_memory_001",
            npc_id="lucas_001",
            content="The player helped Lucas yesterday.",
            timestamp=100,
            importance=0.9,
            event_id="lucas_event_001"
        ),
        Memory(
            id="lucas_memory_002",
            npc_id="lucas_001",
            content="Lucas does not completely trust the player yet.",
            timestamp=200,
            importance=0.7,
            event_id="lucas_event_002"
        ),
        Memory(
            id="lucas_memory_003",
            npc_id="lucas_001",
            content="Lucas promised not to reveal the location of the hidden village.",
            timestamp=300,
            importance=1.0,
            event_id="lucas_event_003"
        )
    ]

    # ==============================
    # Jack
    # ==============================

    jack_personality = Personality(
        honesty=0.8,
        courage=0.8,
        intelligence=0.6,
        kindness=0.7,
        suspicion=0.3,
        confidence=0.8
    )

    jack = NPC(
        npc_id="jack_001",
        name="Jack",
        role="energetic young man",
        personality=jack_personality
    )

    jack.add_goal("Have fun")
    jack.add_goal("Find out what is happening")

    jack.add_secret("Jack sometimes acts before thinking.")

    jack_memories = [
        Memory(
            id="jack_memory_001",
            npc_id="jack_001",
            content="Jack met the player near the village gate.",
            timestamp=100,
            importance=0.6,
            event_id="jack_event_001"
        ),
        Memory(
            id="jack_memory_002",
            npc_id="jack_001",
            content="Jack heard a strange rumor about the forest.",
            timestamp=200,
            importance=0.8,
            event_id="jack_event_002"
        ),
        Memory(
            id="jack_memory_003",
            npc_id="jack_001",
            content="Jack wants to investigate the strange rumor with the player.",
            timestamp=300,
            importance=0.9,
            event_id="jack_event_003"
        )
    ]

    # ==============================
    # Create games
    # ==============================

    npcs = [
        ("Sophia", sophia, sophia_memories),
        ("Barney", barney, barney_memories),
        ("Lucas", lucas, lucas_memories),
        ("Jack", jack, jack_memories)
    ]

    query = "Why do you look so nervous?"

    for name, npc, memories in npcs:

        print("\n")
        print("=" * 60)
        print(name)
        print("=" * 60)

        game = Game(npc, memories)

        print("\nPlayer:")
        print(query)

        response = game.ask_npc(query)

        print("\nNPC:")
        print(response)