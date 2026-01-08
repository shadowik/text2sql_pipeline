"""벡터 인덱서."""

from typing import Any

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
        vector_data = [
            {
                "template_id": template.template_id,
                "template_text": template.template_text,
                "description": template.description,
                "tables": template.tables,
                "columns": template.columns,
                "embedding": embedding,
            }
        ]

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
            {
                "template_id": template.template_id,
                "template_text": template.template_text,
                "description": template.description,
                "tables": template.tables,
                "columns": template.columns,
                "embedding": embedding,
            }
            for template, embedding in zip(unique_templates, embeddings)
        ]

        return self._milvus_adapter.insert_vectors(
            self._collection_name, vector_data
        )

