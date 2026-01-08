"""SQL 설명 생성기."""

from typing import Any, Optional, Protocol


DEFAULT_PROMPT_TEMPLATE = "다음 SQL 쿼리를 한국어로 간단히 설명해주세요:\n\n{sql}"


class LLMClient(Protocol):
    """LLM 클라이언트 프로토콜."""

    def invoke(self, message: str) -> str:
        """메시지를 전송하고 응답을 수신."""
        ...


class DescriptionGenerator:
    """SQL 설명 생성기."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: Optional[str] = None,
    ) -> None:
        """생성기 초기화.

        Args:
            llm_client: LLM 클라이언트
            prompt_template: 프롬프트 템플릿 (기본값 사용 시 None)
        """
        self._llm_client = llm_client
        self._prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE

    def generate(self, sql: str) -> str:
        """SQL에 대한 설명을 생성.

        Args:
            sql: 설명을 생성할 SQL 쿼리

        Returns:
            SQL에 대한 자연어 설명
        """
        prompt = self._prompt_template.format(sql=sql)
        return self._llm_client.invoke(prompt)

    def generate_batch(self, sqls: list[str]) -> list[str]:
        """여러 SQL에 대한 설명을 배치로 생성.

        Args:
            sqls: 설명을 생성할 SQL 쿼리 리스트

        Returns:
            SQL에 대한 자연어 설명 리스트
        """
        return [self.generate(sql) for sql in sqls]

    def generate_with_context(
        self, sql: str, context: dict[str, Any]
    ) -> str:
        """테이블/컬럼 정보와 함께 SQL 설명을 생성.

        Args:
            sql: 설명을 생성할 SQL 쿼리
            context: 테이블 및 컬럼 정보

        Returns:
            SQL에 대한 자연어 설명
        """
        context_text = self._build_context_text(context)
        prompt = f"{context_text}\n\n다음 SQL 쿼리를 한국어로 간단히 설명해주세요:\n\n{sql}"
        return self._llm_client.invoke(prompt)

    def _build_context_text(self, context: dict[str, Any]) -> str:
        """컨텍스트 정보를 텍스트로 변환.

        Args:
            context: 테이블 및 컬럼 정보

        Returns:
            포맷팅된 컨텍스트 텍스트
        """
        lines = ["### 테이블 정보"]
        for table in context.get("tables", []):
            table_name = table.get("name", "")
            table_comment = table.get("comment", "")
            lines.append(f"\n테이블: {table_name} ({table_comment})")
            lines.append("컬럼:")
            for column in table.get("columns", []):
                col_name = column.get("name", "")
                col_comment = column.get("comment", "")
                lines.append(f"  - {col_name}: {col_comment}")
        return "\n".join(lines)

