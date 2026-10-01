from retriever import Retriever
from npc import NPC
from personality import Personality
from memory import Memory
import numpy as np

class Policy:
    def __init__(self, npc, embedding_model):
        self.npc = npc
        self.embedding_model = embedding_model

    def decide(self, query, relevant_memories):
        if not relevant_memories:
            return "DEFLECT"
        if self._question_touches_secret(query, relevant_memories) == True:
            print("The question touches a secret.")
            if self.npc.personality.honesty >= 0.8:
                return "OMIT"
            else:
                return "LIE"
        print("The question does not touch a secret.")
        return "TRUTH"
        
    import numpy as np


class Policy:

    def __init__(self, npc, embedding_model):
        self.npc = npc
        self.embedding_model = embedding_model

    def decide(self, query, relevant_memories):

        # 没有相关记忆
        if not relevant_memories:
            return "DEFLECT"

        # 判断问题是否触及秘密
        if self._question_touches_secret(query):

            print("The question touches a secret.")

            if self.npc.personality.honesty >= 0.8:
                return "OMIT"
            else:
                return "LIE"

        print("The question does not touch a secret.")

        return "TRUTH"


    def _question_touches_secret(self, query):

        # 没有 secret
        if not self.npc.secrets:
            return False

        # Query embedding
        query_embedding = self.embedding_model.encode([query])

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        # Secret embeddings
        secret_embeddings = self.embedding_model.encode(
            self.npc.secrets
        )

        secret_embeddings = np.asarray(
            secret_embeddings,
            dtype="float32"
        )

        # Normalize
        query_embedding = query_embedding / (
            np.linalg.norm(
                query_embedding,
                axis=1,
                keepdims=True
            ) + 1e-8
        )

        secret_embeddings = secret_embeddings / (
            np.linalg.norm(
                secret_embeddings,
                axis=1,
                keepdims=True
            ) + 1e-8
        )

        # Cosine similarity
        similarities = np.dot(
            query_embedding,
            secret_embeddings.T
        )[0]

        for secret, similarity in zip(
            self.npc.secrets,
            similarities
        ):

            print(
                f"Secret similarity: "
                f"{similarity:.4f} | "
                f"{secret}"
            )

            if similarity >= 0.5:
                return True

        return False

if __name__ == "__main__":
    personality = Personality(
        honesty=0.5,
        courage=0.5,
        intelligence=0.5,
        kindness=0.5,
        suspicion=0.5,
        confidence=0.5
    )
    npc = NPC(
        npc_id="npc001",
        name="Emma",
        role="Maid",
        personality=personality
    )
    npc.add_secret("I saw Barbie in the rose garden")
    memory = Memory(
        id="m1",
        npc_id="npc001",
        content="I saw Barbie in the rose garden",
        event_id=None,
        timestamp=100,
        importance=0.9
    )
    memories = [memory]
    policy = Policy(npc)
    result = policy.decide(
        "Where did you see Barbie?",
        memories
    )

    print(result)