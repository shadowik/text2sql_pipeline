"""Elasticsearch 어댑터 테스트."""

from unittest.mock import MagicMock, patch

import pytest

from text2sql.adapters.search.es_adapter import ElasticsearchAdapter
from text2sql.core.config import Settings


@pytest.fixture
def es_settings() -> Settings:
    """Elasticsearch 설정 fixture."""
    return Settings(
        es_host="localhost",
        es_port=9200,
        es_index_name="test_index",
    )


class TestElasticsearchAdapterIndex:
    """Elasticsearch 어댑터 인덱스 테스트."""

    def test_should_check_index_exists_returns_true(
        self, es_settings: Settings
    ) -> None:
        """인덱스가 존재하면 True를 반환해야 함."""
        with patch(
            "text2sql.adapters.search.es_adapter.Elasticsearch"
        ) as mock_es_class:
            # Given
            mock_client = MagicMock()
            mock_client.indices.exists.return_value = True
            mock_es_class.return_value = mock_client

            adapter = ElasticsearchAdapter(es_settings)

            # When
            result = adapter.index_exists("test_index")

            # Then
            mock_client.indices.exists.assert_called_once_with(index="test_index")
            assert result is True

    def test_should_check_index_exists_returns_false(
        self, es_settings: Settings
    ) -> None:
        """인덱스가 존재하지 않으면 False를 반환해야 함."""
        with patch(
            "text2sql.adapters.search.es_adapter.Elasticsearch"
        ) as mock_es_class:
            # Given
            mock_client = MagicMock()
            mock_client.indices.exists.return_value = False
            mock_es_class.return_value = mock_client

            adapter = ElasticsearchAdapter(es_settings)

            # When
            result = adapter.index_exists("non_existent_index")

            # Then
            mock_client.indices.exists.assert_called_once_with(index="non_existent_index")
            assert result is False


class TestElasticsearchAdapterInsert:
    """Elasticsearch 어댑터 문서 삽입 테스트."""

    def test_should_insert_document(self, es_settings: Settings) -> None:
        """문서를 삽입해야 함."""
        with patch(
            "text2sql.adapters.search.es_adapter.Elasticsearch"
        ) as mock_es_class:
            # Given
            mock_client = MagicMock()
            mock_client.index.return_value = {"_id": "doc_1", "result": "created"}
            mock_es_class.return_value = mock_client

            adapter = ElasticsearchAdapter(es_settings)
            document = {
                "template_id": "sql_1",
                "template_text": "SELECT * FROM users WHERE id = ?",
                "description": "사용자 조회 쿼리",
            }

            # When
            result = adapter.insert_document("test_index", "doc_1", document)

            # Then
            mock_client.index.assert_called_once_with(
                index="test_index",
                id="doc_1",
                document=document,
            )
            assert result == "doc_1"

    def test_should_insert_document_without_id(self, es_settings: Settings) -> None:
        """ID 없이 문서를 삽입하면 자동 생성된 ID를 반환해야 함."""
        with patch(
            "text2sql.adapters.search.es_adapter.Elasticsearch"
        ) as mock_es_class:
            # Given
            mock_client = MagicMock()
            mock_client.index.return_value = {"_id": "auto_generated_id", "result": "created"}
            mock_es_class.return_value = mock_client

            adapter = ElasticsearchAdapter(es_settings)
            document = {
                "template_text": "SELECT * FROM users",
                "description": "사용자 전체 조회",
            }

            # When
            result = adapter.insert_document("test_index", None, document)

            # Then
            mock_client.index.assert_called_once_with(
                index="test_index",
                id=None,
                document=document,
            )
            assert result == "auto_generated_id"


class TestElasticsearchAdapterSearch:
    """Elasticsearch 어댑터 검색 테스트."""

    def test_should_search_documents_with_bm25(self, es_settings: Settings) -> None:
        """BM25 검색으로 문서를 검색해야 함."""
        with patch(
            "text2sql.adapters.search.es_adapter.Elasticsearch"
        ) as mock_es_class:
            # Given
            mock_client = MagicMock()
            mock_client.search.return_value = {
                "hits": {
                    "total": {"value": 2},
                    "hits": [
                        {
                            "_id": "doc_1",
                            "_score": 1.5,
                            "_source": {
                                "template_text": "SELECT * FROM users",
                                "description": "사용자 조회",
                            },
                        },
                        {
                            "_id": "doc_2",
                            "_score": 1.2,
                            "_source": {
                                "template_text": "SELECT * FROM customers",
                                "description": "고객 조회",
                            },
                        },
                    ],
                }
            }
            mock_es_class.return_value = mock_client

            adapter = ElasticsearchAdapter(es_settings)

            # When
            results = adapter.search("test_index", "사용자 조회", limit=10)

            # Then
            mock_client.search.assert_called_once()
            call_kwargs = mock_client.search.call_args[1]
            assert call_kwargs["index"] == "test_index"
            assert call_kwargs["size"] == 10
            assert len(results) == 2
            assert results[0]["_id"] == "doc_1"
            assert results[0]["_score"] == 1.5

    def test_should_return_empty_list_when_no_results(
        self, es_settings: Settings
    ) -> None:
        """검색 결과가 없으면 빈 리스트를 반환해야 함."""
        with patch(
            "text2sql.adapters.search.es_adapter.Elasticsearch"
        ) as mock_es_class:
            # Given
            mock_client = MagicMock()
            mock_client.search.return_value = {
                "hits": {
                    "total": {"value": 0},
                    "hits": [],
                }
            }
            mock_es_class.return_value = mock_client

            adapter = ElasticsearchAdapter(es_settings)

            # When
            results = adapter.search("test_index", "존재하지 않는 쿼리")

            # Then
            assert results == []

    def test_should_search_with_specific_fields(self, es_settings: Settings) -> None:
        """특정 필드에서만 검색해야 함."""
        with patch(
            "text2sql.adapters.search.es_adapter.Elasticsearch"
        ) as mock_es_class:
            # Given
            mock_client = MagicMock()
            mock_client.search.return_value = {
                "hits": {"total": {"value": 1}, "hits": [{"_id": "doc_1", "_score": 2.0, "_source": {}}]}
            }
            mock_es_class.return_value = mock_client

            adapter = ElasticsearchAdapter(es_settings)

            # When
            results = adapter.search(
                "test_index",
                "사용자",
                fields=["description", "template_text"],
            )

            # Then
            call_kwargs = mock_client.search.call_args[1]
            assert "multi_match" in str(call_kwargs["query"])
            assert len(results) == 1


class TestElasticsearchAdapterBulk:
    """Elasticsearch 어댑터 bulk 삽입 테스트."""

    def test_should_bulk_insert_documents(self, es_settings: Settings) -> None:
        """여러 문서를 bulk로 삽입해야 함."""
        with patch(
            "text2sql.adapters.search.es_adapter.Elasticsearch"
        ) as mock_es_class, patch(
            "text2sql.adapters.search.es_adapter.helpers"
        ) as mock_helpers:
            # Given
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            mock_helpers.bulk.return_value = (3, [])  # (성공 수, 에러 리스트)

            adapter = ElasticsearchAdapter(es_settings)
            documents = [
                {"_id": "doc_1", "template_text": "SELECT * FROM users"},
                {"_id": "doc_2", "template_text": "SELECT * FROM orders"},
                {"_id": "doc_3", "template_text": "SELECT * FROM products"},
            ]

            # When
            success_count, errors = adapter.bulk_insert("test_index", documents)

            # Then
            mock_helpers.bulk.assert_called_once()
            assert success_count == 3
            assert errors == []

    def test_should_return_errors_on_partial_failure(
        self, es_settings: Settings
    ) -> None:
        """일부 문서 삽입 실패 시 에러를 반환해야 함."""
        with patch(
            "text2sql.adapters.search.es_adapter.Elasticsearch"
        ) as mock_es_class, patch(
            "text2sql.adapters.search.es_adapter.helpers"
        ) as mock_helpers:
            # Given
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            mock_helpers.bulk.return_value = (
                2,
                [{"index": {"_id": "doc_3", "error": "mapping error"}}],
            )

            adapter = ElasticsearchAdapter(es_settings)
            documents = [
                {"_id": "doc_1", "template_text": "SELECT * FROM users"},
                {"_id": "doc_2", "template_text": "SELECT * FROM orders"},
                {"_id": "doc_3", "template_text": "invalid"},
            ]

            # When
            success_count, errors = adapter.bulk_insert("test_index", documents)

            # Then
            assert success_count == 2
            assert len(errors) == 1

    def test_should_return_zero_for_empty_documents(
        self, es_settings: Settings
    ) -> None:
        """빈 문서 리스트는 0을 반환해야 함."""
        with patch(
            "text2sql.adapters.search.es_adapter.Elasticsearch"
        ) as mock_es_class:
            # Given
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client

            adapter = ElasticsearchAdapter(es_settings)

            # When
            success_count, errors = adapter.bulk_insert("test_index", [])

            # Then
            assert success_count == 0
            assert errors == []

