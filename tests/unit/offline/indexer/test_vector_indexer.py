"""VectorIndexer 테스트."""

from unittest.mock import MagicMock

import pytest

from text2sql.core.models import SQLTemplate
from text2sql.offline.indexer.vector_indexer import VectorIndexer


class TestVectorIndexer:
    """VectorIndexer 테스트."""

    def test_index_template_should_embed_and_store_to_milvus(self) -> None:
        """12.1 SQLTemplate을 임베딩하고 Milvus에 저장해야 함."""
        # Given
        mock_embedding_service = MagicMock()
        mock_embedding_service.embed.return_value = [0.1] * 1536

        mock_milvus_adapter = MagicMock()
        mock_milvus_adapter.insert_vectors.return_value = [1]

        indexer = VectorIndexer(
            embedding_service=mock_embedding_service,
            milvus_adapter=mock_milvus_adapter,
            collection_name="sql_templates",
        )

        template = SQLTemplate(
            template_id="tmpl_001",
            template_text="SELECT * FROM users WHERE id = ?",
            description="사용자 정보를 조회하는 쿼리",
            tables=["users"],
            columns=["id"],
        )

        # When
        result = indexer.index_template(template)

        # Then
        mock_embedding_service.embed.assert_called_once_with(
            "사용자 정보를 조회하는 쿼리"
        )
        mock_milvus_adapter.insert_vectors.assert_called_once()
        assert result == [1]

    def test_index_batch_should_embed_and_store_multiple_templates(self) -> None:
        """12.2 여러 템플릿을 배치로 임베딩하고 Milvus에 저장해야 함."""
        # Given
        mock_embedding_service = MagicMock()
        mock_embedding_service.embed_batch.return_value = [
            [0.1] * 1536,
            [0.2] * 1536,
        ]

        mock_milvus_adapter = MagicMock()
        mock_milvus_adapter.insert_vectors.return_value = [1, 2]

        indexer = VectorIndexer(
            embedding_service=mock_embedding_service,
            milvus_adapter=mock_milvus_adapter,
            collection_name="sql_templates",
        )

        templates = [
            SQLTemplate(
                template_id="tmpl_001",
                template_text="SELECT * FROM users WHERE id = ?",
                description="사용자 정보 조회",
                tables=["users"],
                columns=["id"],
            ),
            SQLTemplate(
                template_id="tmpl_002",
                template_text="SELECT * FROM orders WHERE user_id = ?",
                description="주문 정보 조회",
                tables=["orders"],
                columns=["user_id"],
            ),
        ]

        # When
        result = indexer.index_batch(templates)

        # Then
        mock_embedding_service.embed_batch.assert_called_once_with(
            ["사용자 정보 조회", "주문 정보 조회"]
        )
        mock_milvus_adapter.insert_vectors.assert_called_once()
        assert result == [1, 2]

    def test_index_batch_should_skip_duplicate_templates(self) -> None:
        """12.3 중복된 template_id가 있으면 스킵해야 함."""
        # Given
        mock_embedding_service = MagicMock()
        mock_embedding_service.embed_batch.return_value = [[0.1] * 1536]

        mock_milvus_adapter = MagicMock()
        mock_milvus_adapter.insert_vectors.return_value = [1]

        indexer = VectorIndexer(
            embedding_service=mock_embedding_service,
            milvus_adapter=mock_milvus_adapter,
            collection_name="sql_templates",
        )

        templates = [
            SQLTemplate(
                template_id="tmpl_001",
                template_text="SELECT * FROM users WHERE id = ?",
                description="사용자 정보 조회",
                tables=["users"],
                columns=["id"],
            ),
            SQLTemplate(
                template_id="tmpl_001",  # 중복 template_id
                template_text="SELECT * FROM users WHERE id = ?",
                description="사용자 정보 조회 (중복)",
                tables=["users"],
                columns=["id"],
            ),
        ]

        # When
        result = indexer.index_batch(templates)

        # Then - 첫 번째 템플릿만 인덱싱되어야 함
        mock_embedding_service.embed_batch.assert_called_once_with(
            ["사용자 정보 조회"]
        )
        assert result == [1]

