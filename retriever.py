import faiss
import numpy as np
from embedding import Embedding
from memory import Memory

class Retriever:

    def __init__(self, memories, embedding_model):
        self.memories = memories
        self.embedding_model = embedding_model

        self.index = None

        self._build_index()

    def _build_index(self):
        """
        Build FAISS index from current memories.
        """

        if not self.memories:
            self.index = None
            return

        # Get text from each memory
        texts = [
            memory.content
            for memory in self.memories
        ]

        # Convert text to embeddings
        embeddings = self.embedding_model.encode(texts)

        # FAISS requires float32
        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        # Normalize vectors
        # After normalization, Inner Product = Cosine Similarity
        faiss.normalize_L2(embeddings)

        # Embedding dimension
        dimension = embeddings.shape[1]

        # Create FAISS index
        self.index = faiss.IndexFlatIP(dimension)

        # Add memory vectors
        self.index.add(embeddings)

    def retrieve(self, query, top_k=5):
        """
        Retrieve the most semantically relevant memories.

        Returns:
            [
                (memory, similarity_score),
                ...
            ]
        """

        if self.index is None:
            return []

        if not self.memories:
            return []

        # Make sure top_k is valid
        top_k = min(top_k, len(self.memories))

        if top_k <= 0:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode(
            [query]
        )

        # FAISS requires float32
        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        # Normalize query vector
        faiss.normalize_L2(query_embedding)

        # Search FAISS
        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            # FAISS can return -1 when there are not enough results
            if index < 0:
                continue

            memory = self.memories[index]

            results.append(
                (memory, float(score))
            )

        return results


if __name__ == "__main__":

    # Create embedding model
    embedding_model = Embedding()

    # Create test memories
    memories = [

        Memory(
            id="m1",
            npc_id="emma",
            content="I saw Barbie in the rose garden.",
            event_id=None,
            timestamp=100,
            importance=0.9
        ),

        Memory(
            id="m2",
            npc_id="emma",
            content="Barbie was dancing in the ballroom.",
            event_id=None,
            timestamp=110,
            importance=0.7
        ),

        Memory(
            id="m3",
            npc_id="emma",
            content="The prince was talking with the queen.",
            event_id=None,
            timestamp=120,
            importance=0.6
        ),

        Memory(
            id="m4",
            npc_id="emma",
            content="I was cleaning the royal bedroom.",
            event_id=None,
            timestamp=130,
            importance=0.5
        )
    ]

    # Create retriever
    retriever = Retriever(
        memories,
        embedding_model
    )

    # Test queries
    queries = [
        "Where was Barbie?",
        "Where did the princess go?",
        "Who was talking with the queen?"
    ]

    for query in queries:

        print("\n==============================")
        print("Query:", query)
        print("==============================")

        results = retriever.retrieve(
            query,
            top_k=3
        )

        for memory, score in results:

            print(
                f"Score: {score:.4f} | "
                f"Memory: {memory.content}"
            )