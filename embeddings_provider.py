from typing import Iterable, List

from sentence_transformers import SentenceTransformer


class LocalEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode(
            [text],
            normalize_embeddings=True,
        )[0]
        return vector.tolist()

    def __call__(self, texts):
        """Compatibility wrapper: allow the instance to be called like a function.

        - If passed a single string, return the query embedding (list[float]).
        - If passed an iterable of strings, return document embeddings (list[list[float]]).
        This makes the class compatible with code that expects a callable embedder.
        """
        if isinstance(texts, str):
            return self.embed_query(texts)

        # Treat other iterables as documents
        return self.embed_documents(list(texts))


HFInferenceEmbeddings = LocalEmbeddings