from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.documents import Document

from config.settings import AppSettings


@dataclass(frozen=True)
class RerankedDocument:
    document: Document
    score: float | None


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int,
    ) -> list[RerankedDocument]:
        ...


class LocalQwen3Reranker:
    def __init__(
        self,
        model_path: str,
        instruction: str,
        max_length: int = 8192,
        batch_size: int = 4,
        device: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_path = model_path
        self.instruction = instruction
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.trust_remote_code = trust_remote_code
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._token_true_id: int | None = None
        self._token_false_id: int | None = None
        self._prefix_tokens: list[int] | None = None
        self._suffix_tokens: list[int] | None = None

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                padding_side="left",
                trust_remote_code=self.trust_remote_code,
            )
        return self._tokenizer

    @property
    def model(self) -> Any:
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM

            kwargs: dict[str, Any] = {
                "trust_remote_code": self.trust_remote_code,
            }
            if self.device:
                kwargs["device_map"] = {"": self.device}
            elif torch.cuda.is_available():
                kwargs["device_map"] = "auto"
            self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs).eval()
        return self._model

    @property
    def token_true_id(self) -> int:
        if self._token_true_id is None:
            self._token_true_id = int(self.tokenizer.convert_tokens_to_ids("yes"))
        return self._token_true_id

    @property
    def token_false_id(self) -> int:
        if self._token_false_id is None:
            self._token_false_id = int(self.tokenizer.convert_tokens_to_ids("no"))
        return self._token_false_id

    @property
    def prefix_tokens(self) -> list[int]:
        if self._prefix_tokens is None:
            prefix = (
                "<|im_start|>system\n"
                "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
                'Note that the answer can only be "yes" or "no".'
                "<|im_end|>\n<|im_start|>user\n"
            )
            self._prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        return self._prefix_tokens

    @property
    def suffix_tokens(self) -> list[int]:
        if self._suffix_tokens is None:
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self._suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        return self._suffix_tokens

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int,
    ) -> list[RerankedDocument]:
        if top_n <= 0:
            return [RerankedDocument(document=document, score=None) for document in documents]
        if not documents:
            return []

        scores: list[float] = []
        for index in range(0, len(documents), self.batch_size):
            batch = documents[index : index + self.batch_size]
            scores.extend(self._score_batch(query, batch))

        ranked = sorted(
            (
                RerankedDocument(document=document, score=score)
                for document, score in zip(documents, scores, strict=True)
            ),
            key=lambda item: item.score if item.score is not None else 0.0,
            reverse=True,
        )
        return ranked[: min(top_n, len(ranked))]

    def _format_instruction(self, query: str, document: Document) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document.page_content}"
        )

    def _score_batch(self, query: str, documents: list[Document]) -> list[float]:
        import torch

        pairs = [self._format_instruction(query, document) for document in documents]
        max_content_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if max_content_length <= 0:
            raise RuntimeError("LOCAL_RERANK_MAX_LENGTH is too small for Qwen3 reranker prompt tokens")

        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_content_length,
        )
        for index, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = self.prefix_tokens + input_ids + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            yes_no_scores = torch.stack([false_vector, true_vector], dim=1)
            normalized = torch.nn.functional.log_softmax(yes_no_scores, dim=1)
            return normalized[:, 1].exp().detach().cpu().tolist()


class DashScopeReranker:
    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int,
    ) -> list[RerankedDocument]:
        if top_n <= 0:
            return [RerankedDocument(document=document, score=None) for document in documents]
        if not documents:
            return []
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for rerank")

        import dashscope

        response = dashscope.TextReRank.call(
            api_key=self.api_key,
            model=self.model,
            query=query,
            documents=[document.page_content for document in documents],
            top_n=min(top_n, len(documents)),
            return_documents=False,
        )
        status_code = getattr(response, "status_code", 200)
        if status_code != 200:
            message = getattr(response, "message", "DashScope rerank request failed")
            raise RuntimeError(f"DashScope rerank failed: {message}")

        output = getattr(response, "output", None) or response.get("output", {})
        results = output.get("results", [])
        ranked: list[RerankedDocument] = []
        for item in results:
            index = int(item.get("index", item.get("document_index", -1)))
            if 0 <= index < len(documents):
                score = item.get("relevance_score", item.get("score"))
                ranked.append(
                    RerankedDocument(
                        document=documents[index],
                        score=float(score) if score is not None else None,
                    )
                )
        if not ranked:
            raise RuntimeError("DashScope rerank returned no usable results")
        return ranked


def build_reranker(settings: AppSettings) -> Reranker:
    if settings.rerank_provider == "local":
        if not settings.local_rerank_model_path:
            raise ValueError("LOCAL_RERANK_MODEL_PATH is required when RERANK_PROVIDER=local")
        return LocalQwen3Reranker(
            model_path=settings.local_rerank_model_path,
            instruction=settings.local_rerank_instruction,
            max_length=settings.local_rerank_max_length,
            batch_size=settings.local_rerank_batch_size,
            device=settings.local_rerank_device,
            trust_remote_code=settings.local_rerank_trust_remote_code,
        )
    return DashScopeReranker(
        api_key=settings.dashscope_api_key,
        model=settings.qwen_rerank_model,
    )
