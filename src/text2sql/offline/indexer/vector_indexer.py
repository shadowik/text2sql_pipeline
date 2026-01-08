"""벡터 인덱서."""

from typing import Any

from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility

from text2sql.core.models import SQLTemplate


class VectorIndexer:
    """SQL 템플릿을 벡터 저장소에 인덱싱하는 서비스."""

    def __init__(
        self,
        embedding_service: Any,
        milvus_adapter: Any,
        collection_name: str,
    ) -> None:
        """인덱서 초기화.

        Args:
            embedding_service: 임베딩 서비스
            milvus_adapter: Milvus 어댑터
            collection_name: 컬렉션 이름
        """
        self._embedding_service = embedding_service
        self._milvus_adapter = milvus_adapter
        self._collection_name = collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """컬렉션이 없으면 생성."""
        if not utility.has_collection(self._collection_name):
            # 스키마 정의
            # max_length는 바이트 수 기준 (한글 UTF-8은 문자당 3바이트)
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="template_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="template_text", dtype=DataType.VARCHAR, max_length=12000),  # 4000자 * 3바이트
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=6000),  # 2000자 * 3바이트
                FieldSchema(name="tables", dtype=DataType.VARCHAR, max_length=3000),  # 1000자 * 3바이트
                FieldSchema(name="columns", dtype=DataType.VARCHAR, max_length=6000),  # 2000자 * 3바이트
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._embedding_service.dimension),
            ]
            schema = CollectionSchema(fields=fields, description="SQL 템플릿 벡터 저장소")
            collection = Collection(name=self._collection_name, schema=schema)
            
            # 인덱스 생성
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()

    def _prepare_vector_data(self, template: SQLTemplate, embedding: list[float]) -> dict:
        """벡터 데이터를 준비."""
        return {
            "template_id": template.template_id,
            "template_text": template.template_text[:4000],  # max_length 제한
            "description": template.description[:2000],  # max_length 제한
            "tables": ",".join(template.tables)[:1000],  # 리스트를 문자열로 변환
            "columns": ",".join(template.columns)[:2000],  # 리스트를 문자열로 변환
            "embedding": embedding,
        }

    def index_template(self, template: SQLTemplate) -> list[int]:
        """단일 템플릿을 인덱싱.

        Args:
            template: 인덱싱할 SQL 템플릿

        Returns:
            삽입된 레코드의 primary key 리스트
        """
        # 설명을 임베딩
        embedding = self._embedding_service.embed(template.description)

        # Milvus에 저장
        vector_data = [self._prepare_vector_data(template, embedding)]

        return self._milvus_adapter.insert_vectors(
            self._collection_name, vector_data
        )

    def index_batch(self, templates: list[SQLTemplate]) -> list[int]:
        """여러 템플릿을 배치로 인덱싱.

        Args:
            templates: 인덱싱할 SQL 템플릿 리스트

        Returns:
            삽입된 레코드의 primary key 리스트
        """
        if not templates:
            return []

        # 중복 template_id 제거 (첫 번째만 유지)
        seen_ids: set[str] = set()
        unique_templates: list[SQLTemplate] = []
        for template in templates:
            if template.template_id not in seen_ids:
                seen_ids.add(template.template_id)
                unique_templates.append(template)

        if not unique_templates:
            return []

        # 설명들을 배치로 임베딩
        descriptions = [t.description for t in unique_templates]
        embeddings = self._embedding_service.embed_batch(descriptions)

        # Milvus에 저장
        vector_data = [
            self._prepare_vector_data(template, embedding)
            for template, embedding in zip(unique_templates, embeddings)
        ]

        return self._milvus_adapter.insert_vectors(
            self._collection_name, vector_data
        )

