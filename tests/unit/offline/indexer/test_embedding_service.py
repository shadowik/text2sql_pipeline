"""임베딩 서비스 테스트."""

from unittest.mock import MagicMock, patch

import pytest

from text2sql.offline.indexer.embedding_service import EmbeddingService
from text2sql.core.config import Settings


@pytest.fixture
def embedding_settings() -> Settings:
    """임베딩 설정 fixture."""
    return Settings(
        openai_api_key="test-api-key",
    )


class TestEmbeddingServiceSingle:
    """단일 텍스트 임베딩 테스트."""

    def test_should_embed_single_text(self, embedding_settings: Settings) -> None:
        """단일 텍스트를 임베딩해야 함."""
        with patch(
            "text2sql.offline.indexer.embedding_service.OpenAIEmbeddings"
        ) as mock_embeddings_class:
            # Given
            mock_embeddings = MagicMock()
            mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3] * 512  # 1536 차원
            mock_embeddings_class.return_value = mock_embeddings

            service = EmbeddingService(embedding_settings)

            # When
            result = service.embed("사용자 테이블에서 이름을 조회합니다.")

            # Then
            mock_embeddings.embed_query.assert_called_once_with(
                "사용자 테이블에서 이름을 조회합니다."
            )
            assert len(result) == 1536


class TestEmbeddingServiceBatch:
    """배치 텍스트 임베딩 테스트."""

    def test_should_embed_batch_texts(self, embedding_settings: Settings) -> None:
        """여러 텍스트를 배치로 임베딩해야 함."""
        with patch(
            "text2sql.offline.indexer.embedding_service.OpenAIEmbeddings"
        ) as mock_embeddings_class:
            # Given
            mock_embeddings = MagicMock()
            mock_embeddings.embed_documents.return_value = [
                [0.1, 0.2, 0.3] * 512,  # 1536 차원
                [0.4, 0.5, 0.6] * 512,
                [0.7, 0.8, 0.9] * 512,
            ]
            mock_embeddings_class.return_value = mock_embeddings

            service = EmbeddingService(embedding_settings)
            texts = [
                "사용자 테이블에서 이름을 조회합니다.",
                "주문 테이블에서 총액을 조회합니다.",
                "상품 테이블에서 가격을 조회합니다.",
            ]

            # When
            results = service.embed_batch(texts)

            # Then
            mock_embeddings.embed_documents.assert_called_once_with(texts)
            assert len(results) == 3
            assert all(len(r) == 1536 for r in results)


class TestEmbeddingServiceDimension:
    """임베딩 차원 검증 테스트."""

    def test_should_return_expected_dimension(
        self, embedding_settings: Settings
    ) -> None:
        """예상된 임베딩 차원을 반환해야 함."""
        with patch(
            "text2sql.offline.indexer.embedding_service.OpenAIEmbeddings"
        ) as mock_embeddings_class:
            # Given
            mock_embeddings_class.return_value = MagicMock()

            service = EmbeddingService(embedding_settings)

            # When
            dimension = service.dimension

            # Then
            assert dimension == 1536  # OpenAI text-embedding-ada-002 기본 차원

    def test_should_allow_custom_dimension(self, embedding_settings: Settings) -> None:
        """사용자 정의 차원을 설정할 수 있어야 함."""
        with patch(
            "text2sql.offline.indexer.embedding_service.OpenAIEmbeddings"
        ) as mock_embeddings_class:
            # Given
            mock_embeddings_class.return_value = MagicMock()

            service = EmbeddingService(embedding_settings, dimension=3072)

            # When
            dimension = service.dimension

            # Then
            assert dimension == 3072

