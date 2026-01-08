"""임베딩 서비스 (OpenAI 호환 - LM Studio 등 지원)."""

import httpx
from typing import Any

from text2sql.core.config import Settings

DEFAULT_DIMENSION = 1024  # qwen3-embedding-0.6b 기본 차원


class EmbeddingService:
    """텍스트 임베딩 서비스 (OpenAI 호환 API 지원)."""

    def __init__(self, settings: Settings, dimension: int = DEFAULT_DIMENSION) -> None:
        """서비스 초기화.

        Args:
            settings: 애플리케이션 설정
            dimension: 임베딩 벡터 차원 (기본값: 768)
        """
        self._settings = settings
        self._dimension = dimension
        self._base_url = settings.embedding_base_url.rstrip("/")
        self._api_key = settings.embedding_api_key
        self._model = settings.embedding_model

    @property
    def dimension(self) -> int:
        """임베딩 벡터 차원을 반환."""
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """단일 텍스트를 임베딩.

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터
        """
        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "input": text,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """여러 텍스트를 배치로 임베딩.

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "input": texts,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        # 결과를 인덱스 순서대로 정렬
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

