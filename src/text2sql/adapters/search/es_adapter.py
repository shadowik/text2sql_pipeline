"""Elasticsearch 어댑터."""

from typing import Any

from elasticsearch import Elasticsearch, helpers

from text2sql.core.config import Settings


class ElasticsearchAdapter:
    """Elasticsearch 어댑터."""

    def __init__(self, settings: Settings) -> None:
        """어댑터 초기화.

        Args:
            settings: 애플리케이션 설정
        """
        self._settings = settings
        self._client = Elasticsearch(
            hosts=[f"http://{settings.es_host}:{settings.es_port}"]
        )

    def index_exists(self, index_name: str) -> bool:
        """인덱스 존재 여부 확인.

        Args:
            index_name: 확인할 인덱스 이름

        Returns:
            인덱스 존재 여부
        """
        return self._client.indices.exists(index=index_name)

    def insert_document(
        self, index_name: str, doc_id: str | None, document: dict[str, Any]
    ) -> str:
        """문서 삽입.

        Args:
            index_name: 인덱스 이름
            doc_id: 문서 ID (None이면 자동 생성)
            document: 삽입할 문서

        Returns:
            삽입된 문서 ID
        """
        result = self._client.index(
            index=index_name,
            id=doc_id,
            document=document,
        )
        return result["_id"]

    def search(
        self,
        index_name: str,
        query: str,
        limit: int = 10,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 검색 수행.

        Args:
            index_name: 인덱스 이름
            query: 검색 쿼리
            limit: 반환할 최대 결과 수
            fields: 검색할 필드 목록 (None이면 전체 필드)

        Returns:
            검색 결과 리스트
        """
        if fields:
            search_query = {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                }
            }
        else:
            search_query = {
                "multi_match": {
                    "query": query,
                    "fields": ["*"],
                }
            }

        result = self._client.search(
            index=index_name,
            query=search_query,
            size=limit,
        )

        return result["hits"]["hits"]

    def bulk_insert(
        self, index_name: str, documents: list[dict[str, Any]]
    ) -> tuple[int, list[dict[str, Any]]]:
        """여러 문서를 bulk로 삽입.

        Args:
            index_name: 인덱스 이름
            documents: 삽입할 문서 리스트 (_id 필드가 있으면 ID로 사용)

        Returns:
            (성공한 문서 수, 에러 리스트) 튜플
        """
        if not documents:
            return 0, []

        actions = []
        for doc in documents:
            action = {
                "_index": index_name,
                "_source": {k: v for k, v in doc.items() if k != "_id"},
            }
            if "_id" in doc:
                action["_id"] = doc["_id"]
            actions.append(action)

        success_count, errors = helpers.bulk(
            self._client,
            actions,
            raise_on_error=False,
        )

        return success_count, errors

