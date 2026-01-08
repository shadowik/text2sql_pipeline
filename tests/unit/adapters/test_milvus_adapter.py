"""Milvus 어댑터 테스트."""

from unittest.mock import MagicMock, patch

import pytest

from text2sql.adapters.vector_store.milvus_adapter import MilvusAdapter
from text2sql.core.config import Settings


@pytest.fixture
def milvus_settings() -> Settings:
    """Milvus 설정 fixture."""
    return Settings(
        milvus_host="localhost",
        milvus_port=19530,
        milvus_collection_name="test_collection",
    )


class TestMilvusAdapterCollection:
    """Milvus 어댑터 컬렉션 테스트."""

    def test_should_check_collection_exists_returns_true(
        self, milvus_settings: Settings
    ) -> None:
        """컬렉션이 존재하면 True를 반환해야 함."""
        with patch(
            "text2sql.adapters.vector_store.milvus_adapter.utility"
        ) as mock_utility:
            # Given
            mock_utility.has_collection.return_value = True

            adapter = MilvusAdapter(milvus_settings)

            # When
            result = adapter.collection_exists("test_collection")

            # Then
            mock_utility.has_collection.assert_called_once_with("test_collection")
            assert result is True

    def test_should_check_collection_exists_returns_false(
        self, milvus_settings: Settings
    ) -> None:
        """컬렉션이 존재하지 않으면 False를 반환해야 함."""
        with patch(
            "text2sql.adapters.vector_store.milvus_adapter.utility"
        ) as mock_utility:
            # Given
            mock_utility.has_collection.return_value = False

            adapter = MilvusAdapter(milvus_settings)

            # When
            result = adapter.collection_exists("non_existent_collection")

            # Then
            mock_utility.has_collection.assert_called_once_with("non_existent_collection")
            assert result is False


class TestMilvusAdapterInsert:
    """Milvus 어댑터 벡터 삽입 테스트."""

    def test_should_insert_vectors_into_collection(
        self, milvus_settings: Settings
    ) -> None:
        """벡터를 컬렉션에 삽입해야 함."""
        with patch(
            "text2sql.adapters.vector_store.milvus_adapter.Collection"
        ) as mock_collection_class:
            # Given
            mock_collection = MagicMock()
            mock_collection.insert.return_value = MagicMock(primary_keys=[1, 2, 3])
            mock_collection_class.return_value = mock_collection

            adapter = MilvusAdapter(milvus_settings)
            
            vectors = [
                {"id": "sql_1", "embedding": [0.1, 0.2, 0.3], "text": "SELECT * FROM users"},
                {"id": "sql_2", "embedding": [0.4, 0.5, 0.6], "text": "SELECT * FROM orders"},
                {"id": "sql_3", "embedding": [0.7, 0.8, 0.9], "text": "SELECT * FROM products"},
            ]

            # When
            result = adapter.insert_vectors("test_collection", vectors)

            # Then
            mock_collection_class.assert_called_once_with("test_collection")
            mock_collection.insert.assert_called_once()
            assert result == [1, 2, 3]

    def test_should_insert_empty_vectors_returns_empty_list(
        self, milvus_settings: Settings
    ) -> None:
        """빈 벡터 리스트를 삽입하면 빈 리스트를 반환해야 함."""
        adapter = MilvusAdapter(milvus_settings)

        # When
        result = adapter.insert_vectors("test_collection", [])

        # Then
        assert result == []


class TestMilvusAdapterSearch:
    """Milvus 어댑터 유사도 검색 테스트."""

    def test_should_search_similar_vectors(self, milvus_settings: Settings) -> None:
        """유사한 벡터를 검색해야 함."""
        with patch(
            "text2sql.adapters.vector_store.milvus_adapter.Collection"
        ) as mock_collection_class:
            # Given
            mock_hit1 = MagicMock()
            mock_hit1.id = 1
            mock_hit1.distance = 0.1
            mock_hit1.entity.get.side_effect = lambda k: {"text": "SELECT * FROM users"}.get(k)

            mock_hit2 = MagicMock()
            mock_hit2.id = 2
            mock_hit2.distance = 0.2
            mock_hit2.entity.get.side_effect = lambda k: {"text": "SELECT * FROM orders"}.get(k)

            mock_collection = MagicMock()
            mock_collection.search.return_value = [[mock_hit1, mock_hit2]]
            mock_collection_class.return_value = mock_collection

            adapter = MilvusAdapter(milvus_settings)
            query_vector = [0.1, 0.2, 0.3]

            # When
            results = adapter.search(
                collection_name="test_collection",
                query_vector=query_vector,
                limit=2,
                output_fields=["text"],
            )

            # Then
            mock_collection.search.assert_called_once()
            assert len(results) == 2
            assert results[0]["id"] == 1
            assert results[0]["distance"] == 0.1
            assert results[1]["id"] == 2

    def test_should_return_empty_list_when_no_results(
        self, milvus_settings: Settings
    ) -> None:
        """검색 결과가 없으면 빈 리스트를 반환해야 함."""
        with patch(
            "text2sql.adapters.vector_store.milvus_adapter.Collection"
        ) as mock_collection_class:
            # Given
            mock_collection = MagicMock()
            mock_collection.search.return_value = [[]]
            mock_collection_class.return_value = mock_collection

            adapter = MilvusAdapter(milvus_settings)

            # When
            results = adapter.search(
                collection_name="test_collection",
                query_vector=[0.1, 0.2, 0.3],
                limit=10,
            )

            # Then
            assert results == []


class TestMilvusAdapterDelete:
    """Milvus 어댑터 벡터 삭제 테스트."""

    def test_should_delete_vectors_by_ids(self, milvus_settings: Settings) -> None:
        """ID로 벡터를 삭제해야 함."""
        with patch(
            "text2sql.adapters.vector_store.milvus_adapter.Collection"
        ) as mock_collection_class:
            # Given
            mock_collection = MagicMock()
            mock_collection.delete.return_value = MagicMock(delete_count=3)
            mock_collection_class.return_value = mock_collection

            adapter = MilvusAdapter(milvus_settings)
            ids_to_delete = [1, 2, 3]

            # When
            result = adapter.delete_vectors("test_collection", ids_to_delete)

            # Then
            mock_collection_class.assert_called_once_with("test_collection")
            mock_collection.delete.assert_called_once()
            assert result == 3

    def test_should_delete_vectors_by_expression(
        self, milvus_settings: Settings
    ) -> None:
        """표현식으로 벡터를 삭제해야 함."""
        with patch(
            "text2sql.adapters.vector_store.milvus_adapter.Collection"
        ) as mock_collection_class:
            # Given
            mock_collection = MagicMock()
            mock_collection.delete.return_value = MagicMock(delete_count=5)
            mock_collection_class.return_value = mock_collection

            adapter = MilvusAdapter(milvus_settings)

            # When
            result = adapter.delete_by_expr(
                "test_collection", "created_at < '2024-01-01'"
            )

            # Then
            mock_collection.delete.assert_called_once_with("created_at < '2024-01-01'")
            assert result == 5

    def test_should_return_zero_when_no_vectors_deleted(
        self, milvus_settings: Settings
    ) -> None:
        """삭제된 벡터가 없으면 0을 반환해야 함."""
        with patch(
            "text2sql.adapters.vector_store.milvus_adapter.Collection"
        ) as mock_collection_class:
            # Given
            mock_collection = MagicMock()
            mock_collection.delete.return_value = MagicMock(delete_count=0)
            mock_collection_class.return_value = mock_collection

            adapter = MilvusAdapter(milvus_settings)

            # When
            result = adapter.delete_vectors("test_collection", [999])

            # Then
            assert result == 0

