"""OpenAI 호환 LLM 클라이언트 (LM Studio 등 지원)."""

from typing import Any, Iterator

from langchain_openai import ChatOpenAI

from text2sql.core.config import Settings


class RateLimitError(Exception):
    """Rate limit 에러."""

    pass


class TimeoutError(Exception):
    """Timeout 에러."""

    pass


class OpenAIClient:
    """OpenAI 호환 LLM 클라이언트 (LM Studio 등 지원)."""

    def __init__(self, settings: Settings) -> None:
        """클라이언트 초기화.

        Args:
            settings: 애플리케이션 설정
        """
        self._settings = settings
        self._llm: Any = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )

    def invoke(self, message: str) -> str:
        """메시지를 전송하고 응답을 수신.

        Args:
            message: 전송할 메시지

        Returns:
            LLM의 응답 텍스트

        Raises:
            RateLimitError: Rate limit 초과 시
            TimeoutError: 요청 타임아웃 시
        """
        try:
            response = self._llm.invoke(message)
            return response.content
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                raise RateLimitError(f"Rate limit exceeded: {e}") from e
            if "timed out" in error_msg or "timeout" in error_msg:
                raise TimeoutError(f"Request timed out: {e}") from e
            raise

    def stream(self, message: str) -> Iterator[str]:
        """메시지를 전송하고 스트리밍 응답을 수신.

        Args:
            message: 전송할 메시지

        Yields:
            LLM의 응답 청크
        """
        for chunk in self._llm.stream(message):
            yield chunk.content

