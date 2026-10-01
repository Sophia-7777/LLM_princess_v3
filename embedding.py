from sentence_transformers import SentenceTransformer

class Embedding:
    def __init__(self, model_name = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, text: str):
        embeddings = self.model.encode(text, convert_to_numpy=True)
        return embeddings
    

# if __name__ == "__main__":

#     model = Embedding()

#     texts = [
#         "I saw Barbie in the rose garden.",
#         "Barbie was in the ballroom.",
#         "The prince talked to Barbie."
#     ]

#     embeddings = model.encode(texts)

#     print("Embedding shape:")
#     print(embeddings.shape)

#     print("\nFirst embedding:")
#     print(embeddings[0])

if __name__ == "__main__":

    model = Embedding()

    memories = [
        "I saw Barbie in the rose garden.",
        "Barbie was in the ballroom.",
        "The prince talked to Barbie."
    ]

    queries = [
        "Where was Barbie?",
        "Who did the prince talk to?"
    ]

    memory_embeddings = model.encode(memories)
    query_embeddings = model.encode(queries)

    for i, query in enumerate(queries):

        print("\nQuery:")
        print(query)

        scores = query_embeddings[i] @ memory_embeddings.T

        print("Similarity scores:")
        print(scores)