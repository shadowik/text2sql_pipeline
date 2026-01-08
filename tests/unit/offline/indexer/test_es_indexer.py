"""ESIndexer 테스트."""

from unittest.mock import MagicMock

import pytest

from text2sql.core.models import SQLTemplate
from text2sql.offline.indexer.es_indexer import ESIndexer


class TestESIndexer:
    """ESIndexer 테스트."""

    def test_index_template_should_store_to_elasticsearch(self) -> None:
        """13.1 SQLTemplate을 ES에 저장해야 함."""
        # Given
        mock_es_adapter = MagicMock()
        mock_es_adapter.insert_document.return_value = "tmpl_001"

        indexer = ESIndexer(
            es_adapter=mock_es_adapter,
            index_name="sql_templates",
        )

        template = SQLTemplate(
            template_id="tmpl_001",
            template_text="SELECT * FROM users WHERE id = ?",
            description="사용자 정보를 조회하는 쿼리",
            tables=["users"],
            columns=["id"],
            exec_count=100,
        )

        # When
        result = indexer.index_template(template)

        # Then
        mock_es_adapter.insert_document.assert_called_once_with(
            "sql_templates",
            "tmpl_001",
            {
                "template_id": "tmpl_001",
                "template_text": "SELECT * FROM users WHERE id = ?",
                "description": "사용자 정보를 조회하는 쿼리",
                "tables": ["users"],
                "columns": ["id"],
                "exec_count": 100,
            },
        )
        assert result == "tmpl_001"

    def test_index_batch_should_bulk_store_to_elasticsearch(self) -> None:
        """13.2 여러 템플릿을 bulk로 ES에 저장해야 함."""
        # Given
        mock_es_adapter = MagicMock()
        mock_es_adapter.bulk_insert.return_value = (2, [])

        indexer = ESIndexer(
            es_adapter=mock_es_adapter,
            index_name="sql_templates",
        )

        templates = [
            SQLTemplate(
                template_id="tmpl_001",
                template_text="SELECT * FROM users WHERE id = ?",
                description="사용자 정보 조회",
                tables=["users"],
                columns=["id"],
                exec_count=100,
            ),
            SQLTemplate(
                template_id="tmpl_002",
                template_text="SELECT * FROM orders WHERE user_id = ?",
                description="주문 정보 조회",
                tables=["orders"],
                columns=["user_id"],
                exec_count=50,
            ),
        ]

        # When
        success_count, errors = indexer.index_batch(templates)

        # Then
        mock_es_adapter.bulk_insert.assert_called_once()
        call_args = mock_es_adapter.bulk_insert.call_args
        assert call_args[0][0] == "sql_templates"
        assert len(call_args[0][1]) == 2
        assert success_count == 2
        assert errors == []

    def test_create_index_with_mapping_should_set_correct_mapping(self) -> None:
        """13.3 인덱스 생성 시 올바른 매핑을 설정해야 함."""
        # Given
        mock_es_adapter = MagicMock()
        mock_es_adapter.index_exists.return_value = False

        indexer = ESIndexer(
            es_adapter=mock_es_adapter,
            index_name="sql_templates",
        )

        # When
        indexer.create_index_if_not_exists()

        # Then
        mock_es_adapter.index_exists.assert_called_once_with("sql_templates")
        mock_es_adapter.create_index.assert_called_once()

        # 매핑 검증
        call_args = mock_es_adapter.create_index.call_args
        index_name = call_args[0][0]
        mapping = call_args[0][1]

        assert index_name == "sql_templates"
        assert "properties" in mapping["mappings"]
        assert "template_id" in mapping["mappings"]["properties"]
        assert "template_text" in mapping["mappings"]["properties"]
        assert "description" in mapping["mappings"]["properties"]
        assert "tables" in mapping["mappings"]["properties"]
        assert "columns" in mapping["mappings"]["properties"]

