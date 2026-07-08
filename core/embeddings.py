from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from config.settings import AppSettings


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(
        self,
        model_path: str,
        expected_dimension: int,
        batch_size: int = 16,
        device: str | None = None,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_path = model_path
        self.expected_dimension = expected_dimension
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.trust_remote_code = trust_remote_code
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {
                "trust_remote_code": self.trust_remote_code,
            }
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_path, **kwargs)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result = vectors.tolist()
        self._validate_dimension(result)
        return result

    def _validate_dimension(self, vectors: list[list[float]]) -> None:
        if not vectors:
            return
        actual_dimension = len(vectors[0])
        if actual_dimension != self.expected_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected EMBEDDING_DIMENSION={self.expected_dimension}, "
                f"got {actual_dimension} from {self.model_path}"
            )

    def probe_dimension(self) -> int:
        vector = self.embed_query("dimension probe")
        return len(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]


def build_embeddings(settings: AppSettings) -> Embeddings:
    if settings.embedding_provider != "local":
        raise ValueError("Only local embeddings are supported. Set EMBEDDING_PROVIDER=local")
    if not settings.local_embedding_model_path:
        raise ValueError("LOCAL_EMBEDDING_MODEL_PATH is required when EMBEDDING_PROVIDER=local")
    return LocalSentenceTransformerEmbeddings(
        model_path=settings.local_embedding_model_path,
        expected_dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        device=settings.local_embedding_device,
        trust_remote_code=settings.local_embedding_trust_remote_code,
    )
