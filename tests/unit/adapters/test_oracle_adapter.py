"""Oracle 어댑터 테스트."""

from unittest.mock import MagicMock, patch

import pytest

from text2sql.adapters.database.oracle_adapter import OracleAdapter
from text2sql.core.config import Settings


@pytest.fixture
def oracle_settings() -> Settings:
    """Oracle 설정 fixture."""
    return Settings(
        oracle_host="testhost",
        oracle_port=1521,
        oracle_service_name="TESTDB",
        oracle_user="testuser",
        oracle_password="testpass",
    )


class TestOracleAdapterConnection:
    """Oracle 어댑터 연결 테스트."""

    def test_should_create_connection_with_settings(
        self, oracle_settings: Settings
    ) -> None:
        """설정을 사용하여 연결을 생성해야 함."""
        with patch("text2sql.adapters.database.oracle_adapter.oracledb") as mock_oracledb:
            mock_connection = MagicMock()
            mock_oracledb.connect.return_value = mock_connection

            # When
            adapter = OracleAdapter(oracle_settings)
            connection = adapter.connect()

            # Then
            mock_oracledb.connect.assert_called_once()
            call_kwargs = mock_oracledb.connect.call_args[1]
            assert call_kwargs["user"] == "testuser"
            assert call_kwargs["password"] == "testpass"
            assert "testhost:1521/TESTDB" in call_kwargs["dsn"]
            assert connection == mock_connection


class TestOracleAdapterQuery:
    """Oracle 어댑터 쿼리 실행 테스트."""

    def test_should_execute_select_query_and_return_rows(
        self, oracle_settings: Settings
    ) -> None:
        """SELECT 쿼리를 실행하고 결과를 반환해야 함."""
        with patch("text2sql.adapters.database.oracle_adapter.oracledb") as mock_oracledb:
            # Given
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                (1, "Alice"),
                (2, "Bob"),
            ]
            mock_cursor.description = [
                ("ID", None, None, None, None, None, None),
                ("NAME", None, None, None, None, None, None),
            ]
            mock_connection = MagicMock()
            mock_connection.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_connection.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_oracledb.connect.return_value = mock_connection

            adapter = OracleAdapter(oracle_settings)
            adapter.connect()

            # When
            result = adapter.execute_query("SELECT id, name FROM users")

            # Then
            assert len(result) == 2
            assert result[0] == {"ID": 1, "NAME": "Alice"}
            assert result[1] == {"ID": 2, "NAME": "Bob"}


class TestOracleAdapterConnectionPool:
    """Oracle 어댑터 연결 풀 테스트."""

    def test_should_create_connection_pool(self, oracle_settings: Settings) -> None:
        """연결 풀을 생성해야 함."""
        with patch("text2sql.adapters.database.oracle_adapter.oracledb") as mock_oracledb:
            # Given
            mock_pool = MagicMock()
            mock_oracledb.create_pool.return_value = mock_pool

            # When
            adapter = OracleAdapter(oracle_settings)
            pool = adapter.create_pool(min_connections=2, max_connections=10)

            # Then
            mock_oracledb.create_pool.assert_called_once()
            call_kwargs = mock_oracledb.create_pool.call_args[1]
            assert call_kwargs["user"] == "testuser"
            assert call_kwargs["password"] == "testpass"
            assert call_kwargs["min"] == 2
            assert call_kwargs["max"] == 10
            assert pool == mock_pool

    def test_should_acquire_connection_from_pool(
        self, oracle_settings: Settings
    ) -> None:
        """풀에서 연결을 가져와야 함."""
        with patch("text2sql.adapters.database.oracle_adapter.oracledb") as mock_oracledb:
            # Given
            mock_connection = MagicMock()
            mock_pool = MagicMock()
            mock_pool.acquire.return_value = mock_connection
            mock_oracledb.create_pool.return_value = mock_pool

            adapter = OracleAdapter(oracle_settings)
            adapter.create_pool(min_connections=2, max_connections=10)

            # When
            connection = adapter.acquire_connection()

            # Then
            mock_pool.acquire.assert_called_once()
            assert connection == mock_connection

    def test_should_release_connection_to_pool(
        self, oracle_settings: Settings
    ) -> None:
        """연결을 풀에 반환해야 함."""
        with patch("text2sql.adapters.database.oracle_adapter.oracledb") as mock_oracledb:
            # Given
            mock_connection = MagicMock()
            mock_pool = MagicMock()
            mock_pool.acquire.return_value = mock_connection
            mock_oracledb.create_pool.return_value = mock_pool

            adapter = OracleAdapter(oracle_settings)
            adapter.create_pool(min_connections=2, max_connections=10)
            connection = adapter.acquire_connection()

            # When
            adapter.release_connection(connection)

            # Then
            mock_pool.release.assert_called_once_with(mock_connection)


class TestOracleAdapterTimeout:
    """Oracle 어댑터 타임아웃 테스트."""

    def test_should_handle_query_timeout(self, oracle_settings: Settings) -> None:
        """쿼리 타임아웃 발생 시 적절한 예외를 발생시켜야 함."""
        with patch("text2sql.adapters.database.oracle_adapter.oracledb") as mock_oracledb:
            # Given
            mock_cursor = MagicMock()
            # Oracle 타임아웃 에러 시뮬레이션 (ORA-01013: 사용자 요청에 의해 현재 작업 취소됨)
            mock_cursor.execute.side_effect = Exception("ORA-01013: user requested cancel of current operation")
            mock_connection = MagicMock()
            mock_connection.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_connection.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_oracledb.connect.return_value = mock_connection

            adapter = OracleAdapter(oracle_settings)
            adapter.connect()

            # When / Then
            with pytest.raises(Exception) as exc_info:
                adapter.execute_query("SELECT * FROM large_table", timeout=30)
            
            assert "timeout" in str(exc_info.value).lower() or "ORA-01013" in str(exc_info.value)

    def test_should_execute_query_with_timeout_setting(
        self, oracle_settings: Settings
    ) -> None:
        """타임아웃 설정과 함께 쿼리를 실행해야 함."""
        with patch("text2sql.adapters.database.oracle_adapter.oracledb") as mock_oracledb:
            # Given
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [(1,)]
            mock_cursor.description = [("ID", None, None, None, None, None, None)]
            mock_connection = MagicMock()
            mock_connection.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_connection.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_oracledb.connect.return_value = mock_connection

            adapter = OracleAdapter(oracle_settings)
            adapter.connect()

            # When
            result = adapter.execute_query("SELECT 1 FROM dual", timeout=30)

            # Then
            # 타임아웃 설정을 위한 call_timeout 속성이 설정되어야 함
            assert mock_connection.call_timeout == 30000  # 밀리초 단위
            assert len(result) == 1

