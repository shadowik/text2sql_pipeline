"""임베딩 서비스."""

from typing import Any

from langchain_openai import OpenAIEmbeddings

from text2sql.core.config import Settings

DEFAULT_DIMENSION = 1536  # OpenAI text-embedding-ada-002 기본 차원


class EmbeddingService:
    """텍스트 임베딩 서비스."""

    def __init__(self, settings: Settings, dimension: int = DEFAULT_DIMENSION) -> None:
        """서비스 초기화.

        Args:
            settings: 애플리케이션 설정
            dimension: 임베딩 벡터 차원 (기본값: 1536)
        """
        self._settings = settings
        self._dimension = dimension
        self._embeddings: Any = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
        )

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
        return self._embeddings.embed_query(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """여러 텍스트를 배치로 임베딩.

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        return self._embeddings.embed_documents(texts)

