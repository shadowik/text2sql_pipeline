"""Milvus 벡터 저장소 어댑터."""

from typing import Any

from pymilvus import Collection, utility

from text2sql.core.config import Settings


class MilvusAdapter:
    """Milvus 벡터 저장소 어댑터."""

    def __init__(self, settings: Settings) -> None:
        """어댑터 초기화.

        Args:
            settings: 애플리케이션 설정
        """
        self._settings = settings

    def collection_exists(self, collection_name: str) -> bool:
        """컬렉션 존재 여부 확인.

        Args:
            collection_name: 확인할 컬렉션 이름

        Returns:
            컬렉션 존재 여부
        """
        return utility.has_collection(collection_name)

    def insert_vectors(
        self, collection_name: str, vectors: list[dict[str, Any]]
    ) -> list[int]:
        """벡터를 컬렉션에 삽입.

        Args:
            collection_name: 컬렉션 이름
            vectors: 삽입할 벡터 데이터 리스트

        Returns:
            삽입된 레코드의 primary key 리스트
        """
        if not vectors:
            return []

        collection = Collection(collection_name)
        result = collection.insert(vectors)
        return list(result.primary_keys)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """유사도 검색 수행.

        Args:
            collection_name: 컬렉션 이름
            query_vector: 검색할 벡터
            limit: 반환할 최대 결과 수
            output_fields: 반환할 필드 목록

        Returns:
            검색 결과 리스트
        """
        collection = Collection(collection_name)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=output_fields or [],
        )

        output = []
        for hits in results:
            for hit in hits:
                item = {
                    "id": hit.id,
                    "distance": hit.distance,
                }
                # output_fields에서 추가 필드 가져오기
                if output_fields:
                    for field in output_fields:
                        item[field] = hit.entity.get(field)
                output.append(item)

        return output

    def delete_vectors(self, collection_name: str, ids: list[int]) -> int:
        """ID로 벡터 삭제.

        Args:
            collection_name: 컬렉션 이름
            ids: 삭제할 벡터 ID 리스트

        Returns:
            삭제된 벡터 수
        """
        if not ids:
            return 0

        collection = Collection(collection_name)
        # Milvus는 표현식으로 삭제하므로 ID 리스트를 표현식으로 변환
        expr = f"id in {ids}"
        result = collection.delete(expr)
        return result.delete_count

    def delete_by_expr(self, collection_name: str, expr: str) -> int:
        """표현식으로 벡터 삭제.

        Args:
            collection_name: 컬렉션 이름
            expr: 삭제 조건 표현식

        Returns:
            삭제된 벡터 수
        """
        collection = Collection(collection_name)
        result = collection.delete(expr)
        return result.delete_count

