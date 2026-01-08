"""용어 사전 빌더 - CSV 용어사전을 파싱하고 ES에 인덱싱."""

import csv
from pathlib import Path
from typing import Any

from text2sql.core.models import GlossaryTerm


class GlossaryBuilder:
    """용어 사전을 빌드하는 서비스."""

    def __init__(self, es_adapter: Any = None) -> None:
        """용어 사전 빌더 초기화.

        Args:
            es_adapter: Elasticsearch 어댑터 (옵션)
        """
        self._es_adapter = es_adapter

    def parse_csv(self, csv_path: Path) -> list[dict[str, str]]:
        """CSV 파일을 파싱하여 용어 목록 반환.

        Args:
            csv_path: CSV 파일 경로

        Returns:
            용어 딕셔너리 리스트
        """
        terms = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                terms.append(dict(row))
        return terms

    def create_term(self, term_dict: dict[str, str]) -> GlossaryTerm:
        """딕셔너리에서 GlossaryTerm 객체 생성.

        Args:
            term_dict: 용어 정보 딕셔너리

        Returns:
            GlossaryTerm 객체
        """
        return GlossaryTerm(
            term=term_dict["term"],
            korean_name=term_dict["korean_name"],
            description=term_dict["description"],
            category=term_dict.get("category"),
        )

    def index_terms(
        self, terms: list[GlossaryTerm], index_name: str
    ) -> tuple[int, list[dict[str, Any]]]:
        """용어들을 ES에 인덱싱.

        Args:
            terms: GlossaryTerm 리스트
            index_name: ES 인덱스 이름

        Returns:
            (성공 수, 에러 리스트) 튜플
        """
        if not terms:
            return 0, []

        documents = [
            {
                "_id": term.term,
                "term": term.term,
                "korean_name": term.korean_name,
                "description": term.description,
                "category": term.category,
            }
            for term in terms
        ]

        return self._es_adapter.bulk_insert(index_name, documents)

