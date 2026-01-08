"""OpenAI 클라이언트 테스트."""

from unittest.mock import MagicMock, patch

import pytest

from text2sql.adapters.llm.openai_client import OpenAIClient
from text2sql.core.config import Settings


@pytest.fixture
def openai_settings() -> Settings:
    """OpenAI 설정 fixture."""
    return Settings(
        openai_api_key="test-api-key",
        openai_model="gpt-4o-mini",
    )


class TestOpenAIClientCreation:
    """OpenAI 클라이언트 생성 테스트."""

    def test_should_create_chat_openai_instance(
        self, openai_settings: Settings
    ) -> None:
        """ChatOpenAI 인스턴스를 생성해야 함."""
        with patch("text2sql.adapters.llm.openai_client.ChatOpenAI") as mock_chat_openai:
            mock_instance = MagicMock()
            mock_chat_openai.return_value = mock_instance

            # When
            client = OpenAIClient(openai_settings)

            # Then
            mock_chat_openai.assert_called_once()
            call_kwargs = mock_chat_openai.call_args[1]
            assert call_kwargs["api_key"] == "test-api-key"
            assert call_kwargs["model"] == "gpt-4o-mini"
            assert client._llm == mock_instance


class TestOpenAIClientInvoke:
    """OpenAI 클라이언트 메시지 전송 테스트."""

    def test_should_send_message_and_receive_response(
        self, openai_settings: Settings
    ) -> None:
        """메시지를 전송하고 응답을 수신해야 함."""
        with patch("text2sql.adapters.llm.openai_client.ChatOpenAI") as mock_chat_openai:
            # Given
            mock_response = MagicMock()
            mock_response.content = "이 SQL은 사용자 테이블에서 이름을 조회합니다."
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_chat_openai.return_value = mock_llm

            client = OpenAIClient(openai_settings)

            # When
            result = client.invoke("SQL 설명해주세요: SELECT name FROM users")

            # Then
            mock_llm.invoke.assert_called_once()
            assert result == "이 SQL은 사용자 테이블에서 이름을 조회합니다."


class TestOpenAIClientStream:
    """OpenAI 클라이언트 스트리밍 테스트."""

    def test_should_stream_response(self, openai_settings: Settings) -> None:
        """스트리밍 응답을 반환해야 함."""
        with patch("text2sql.adapters.llm.openai_client.ChatOpenAI") as mock_chat_openai:
            # Given
            mock_chunk1 = MagicMock()
            mock_chunk1.content = "이 SQL은 "
            mock_chunk2 = MagicMock()
            mock_chunk2.content = "사용자 테이블에서 "
            mock_chunk3 = MagicMock()
            mock_chunk3.content = "이름을 조회합니다."

            mock_llm = MagicMock()
            mock_llm.stream.return_value = iter([mock_chunk1, mock_chunk2, mock_chunk3])
            mock_chat_openai.return_value = mock_llm

            client = OpenAIClient(openai_settings)

            # When
            chunks = list(client.stream("SQL 설명해주세요: SELECT name FROM users"))

            # Then
            mock_llm.stream.assert_called_once()
            assert chunks == ["이 SQL은 ", "사용자 테이블에서 ", "이름을 조회합니다."]


class TestOpenAIClientErrorHandling:
    """OpenAI 클라이언트 에러 핸들링 테스트."""

    def test_should_handle_rate_limit_error(self, openai_settings: Settings) -> None:
        """Rate limit 에러를 처리해야 함."""
        with patch("text2sql.adapters.llm.openai_client.ChatOpenAI") as mock_chat_openai:
            # Given
            from text2sql.adapters.llm.openai_client import RateLimitError

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("Rate limit exceeded")
            mock_chat_openai.return_value = mock_llm

            client = OpenAIClient(openai_settings)

            # When / Then
            with pytest.raises(RateLimitError) as exc_info:
                client.invoke("테스트 메시지")
            assert "Rate limit" in str(exc_info.value)

    def test_should_handle_timeout_error(self, openai_settings: Settings) -> None:
        """Timeout 에러를 처리해야 함."""
        with patch("text2sql.adapters.llm.openai_client.ChatOpenAI") as mock_chat_openai:
            # Given
            from text2sql.adapters.llm.openai_client import TimeoutError

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("Request timed out")
            mock_chat_openai.return_value = mock_llm

            client = OpenAIClient(openai_settings)

            # When / Then
            with pytest.raises(TimeoutError) as exc_info:
                client.invoke("테스트 메시지")
            assert "timed out" in str(exc_info.value)

