"""GlossaryBuilder 테스트."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from text2sql.core.models import GlossaryTerm
from text2sql.offline.schema.glossary_builder import GlossaryBuilder


class TestGlossaryBuilder:
    """GlossaryBuilder 테스트 클래스."""

    def test_parse_csv_returns_terms(self) -> None:
        """17.1 CSV 단어사전 파싱 테스트."""
        # Given
        csv_content = """term,korean_name,description,category
USER_ID,사용자ID,고유 사용자 식별자,식별자
USER_NAME,사용자명,사용자의 이름,기본정보
EMAIL,이메일,이메일 주소,연락처"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            glossary_builder = GlossaryBuilder()

            # When
            result = glossary_builder.parse_csv(csv_path)

            # Then
            assert len(result) == 3
            assert result[0]["term"] == "USER_ID"
            assert result[0]["korean_name"] == "사용자ID"
            assert result[1]["term"] == "USER_NAME"
            assert result[2]["category"] == "연락처"
        finally:
            csv_path.unlink()

    def test_create_glossary_term_from_dict(self) -> None:
        """17.2 GlossaryTerm 생성 테스트."""
        # Given
        term_dict = {
            "term": "USER_ID",
            "korean_name": "사용자ID",
            "description": "고유 사용자 식별자",
            "category": "식별자",
        }
        glossary_builder = GlossaryBuilder()

        # When
        result = glossary_builder.create_term(term_dict)

        # Then
        assert isinstance(result, GlossaryTerm)
        assert result.term == "USER_ID"
        assert result.korean_name == "사용자ID"
        assert result.description == "고유 사용자 식별자"
        assert result.category == "식별자"

    def test_index_terms_to_elasticsearch(self) -> None:
        """17.3 ES 인덱싱 테스트."""
        # Given
        mock_es_adapter = MagicMock()
        mock_es_adapter.bulk_insert.return_value = (3, [])
        
        terms = [
            GlossaryTerm(
                term="USER_ID",
                korean_name="사용자ID",
                description="고유 사용자 식별자",
                category="식별자",
            ),
            GlossaryTerm(
                term="USER_NAME",
                korean_name="사용자명",
                description="사용자의 이름",
                category="기본정보",
            ),
            GlossaryTerm(
                term="EMAIL",
                korean_name="이메일",
                description="이메일 주소",
                category="연락처",
            ),
        ]
        
        glossary_builder = GlossaryBuilder(es_adapter=mock_es_adapter)

        # When
        success_count, errors = glossary_builder.index_terms(
            terms, index_name="glossary"
        )

        # Then
        assert success_count == 3
        assert errors == []
        mock_es_adapter.bulk_insert.assert_called_once()
        call_args = mock_es_adapter.bulk_insert.call_args
        assert call_args[0][0] == "glossary"
        assert len(call_args[0][1]) == 3

