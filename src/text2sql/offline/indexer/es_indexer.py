"""ES 인덱서."""

from typing import Any

from text2sql.core.models import SQLTemplate


class ESIndexer:
    """SQL 템플릿을 Elasticsearch에 인덱싱하는 서비스."""

    def __init__(
        self,
        es_adapter: Any,
        index_name: str,
    ) -> None:
        """인덱서 초기화.

        Args:
            es_adapter: Elasticsearch 어댑터
            index_name: 인덱스 이름
        """
        self._es_adapter = es_adapter
        self._index_name = index_name

    def index_template(self, template: SQLTemplate) -> str:
        """단일 템플릿을 인덱싱.

        Args:
            template: 인덱싱할 SQL 템플릿

        Returns:
            삽입된 문서 ID
        """
        document = {
            "template_id": template.template_id,
            "template_text": template.template_text,
            "description": template.description,
            "tables": template.tables,
            "columns": template.columns,
            "exec_count": template.exec_count,
        }

        return self._es_adapter.insert_document(
            self._index_name, template.template_id, document
        )

    def index_batch(
        self, templates: list[SQLTemplate]
    ) -> tuple[int, list[dict[str, Any]]]:
        """여러 템플릿을 배치로 인덱싱.

        Args:
            templates: 인덱싱할 SQL 템플릿 리스트

        Returns:
            (성공 수, 에러 리스트) 튜플
        """
        if not templates:
            return 0, []

        documents = [
            {
                "_id": template.template_id,
                "template_id": template.template_id,
                "template_text": template.template_text,
                "description": template.description,
                "tables": template.tables,
                "columns": template.columns,
                "exec_count": template.exec_count,
            }
            for template in templates
        ]

        return self._es_adapter.bulk_insert(self._index_name, documents)

    def create_index_if_not_exists(self) -> bool:
        """인덱스가 없으면 매핑과 함께 생성.

        Returns:
            인덱스가 생성되었으면 True, 이미 존재하면 False
        """
        if self._es_adapter.index_exists(self._index_name):
            return False

        mapping = {
            "mappings": {
                "properties": {
                    "template_id": {"type": "keyword"},
                    "template_text": {"type": "text"},
                    "description": {"type": "text"},
                    "tables": {"type": "keyword"},
                    "columns": {"type": "keyword"},
                    "exec_count": {"type": "integer"},
                }
            }
        }

        self._es_adapter.create_index(self._index_name, mapping)
        return True

