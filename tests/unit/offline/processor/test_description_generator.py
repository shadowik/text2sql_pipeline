"""설명 생성기 테스트."""

from unittest.mock import MagicMock

import pytest

from text2sql.offline.processor.description_generator import DescriptionGenerator


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM 클라이언트 fixture."""
    return MagicMock()


class TestDescriptionGeneratorSingle:
    """단일 SQL 설명 생성 테스트."""

    def test_should_generate_description_for_single_sql(
        self, mock_llm_client: MagicMock
    ) -> None:
        """단일 SQL에 대한 설명을 생성해야 함."""
        # Given
        mock_llm_client.invoke.return_value = "사용자 테이블에서 이름을 조회하는 쿼리입니다."
        generator = DescriptionGenerator(llm_client=mock_llm_client)
        sql = "SELECT name FROM users WHERE id = :1"

        # When
        result = generator.generate(sql)

        # Then
        mock_llm_client.invoke.assert_called_once()
        assert result == "사용자 테이블에서 이름을 조회하는 쿼리입니다."


class TestDescriptionGeneratorBatch:
    """배치 SQL 설명 생성 테스트."""

    def test_should_generate_descriptions_for_batch_sql(
        self, mock_llm_client: MagicMock
    ) -> None:
        """여러 SQL에 대한 설명을 배치로 생성해야 함."""
        # Given
        mock_llm_client.invoke.side_effect = [
            "사용자 테이블에서 이름을 조회하는 쿼리입니다.",
            "주문 테이블에서 총액을 조회하는 쿼리입니다.",
            "상품 테이블에서 가격을 조회하는 쿼리입니다.",
        ]
        generator = DescriptionGenerator(llm_client=mock_llm_client)
        sqls = [
            "SELECT name FROM users WHERE id = :1",
            "SELECT total FROM orders WHERE order_id = :1",
            "SELECT price FROM products WHERE product_id = :1",
        ]

        # When
        results = generator.generate_batch(sqls)

        # Then
        assert mock_llm_client.invoke.call_count == 3
        assert len(results) == 3
        assert results[0] == "사용자 테이블에서 이름을 조회하는 쿼리입니다."
        assert results[1] == "주문 테이블에서 총액을 조회하는 쿼리입니다."
        assert results[2] == "상품 테이블에서 가격을 조회하는 쿼리입니다."


class TestDescriptionGeneratorPromptTemplate:
    """프롬프트 템플릿 적용 테스트."""

    def test_should_use_custom_prompt_template(
        self, mock_llm_client: MagicMock
    ) -> None:
        """사용자 정의 프롬프트 템플릿을 적용해야 함."""
        # Given
        mock_llm_client.invoke.return_value = "간결한 설명"
        custom_template = "SQL: {sql}\n\n위 쿼리의 목적을 한 줄로 설명하세요."
        generator = DescriptionGenerator(
            llm_client=mock_llm_client,
            prompt_template=custom_template,
        )
        sql = "SELECT name FROM users WHERE id = :1"

        # When
        generator.generate(sql)

        # Then
        call_args = mock_llm_client.invoke.call_args[0][0]
        assert "SQL: SELECT name FROM users WHERE id = :1" in call_args
        assert "한 줄로 설명하세요" in call_args


class TestDescriptionGeneratorWithContext:
    """테이블/컬럼 정보 포함 프롬프트 테스트."""

    def test_should_include_table_and_column_info_in_prompt(
        self, mock_llm_client: MagicMock
    ) -> None:
        """테이블 및 컬럼 정보를 프롬프트에 포함해야 함."""
        # Given
        mock_llm_client.invoke.return_value = "사용자 ID로 사용자 이름을 조회합니다."
        generator = DescriptionGenerator(llm_client=mock_llm_client)
        sql = "SELECT name FROM users WHERE id = :1"
        context = {
            "tables": [
                {
                    "name": "users",
                    "comment": "사용자 정보 테이블",
                    "columns": [
                        {"name": "id", "comment": "사용자 ID"},
                        {"name": "name", "comment": "사용자 이름"},
                    ],
                }
            ]
        }

        # When
        result = generator.generate_with_context(sql, context)

        # Then
        call_args = mock_llm_client.invoke.call_args[0][0]
        assert "users" in call_args
        assert "사용자 정보 테이블" in call_args
        assert "사용자 ID" in call_args
        assert "사용자 이름" in call_args
        assert result == "사용자 ID로 사용자 이름을 조회합니다."

