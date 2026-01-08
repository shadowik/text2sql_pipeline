"""LogCollector 테스트."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from text2sql.core.models import RawSQLLog
from text2sql.offline.ingestor.log_collector import LogCollector


class TestLogCollector:
    """LogCollector 테스트."""

    def test_collect_logs_should_fetch_from_ipa_table(self) -> None:
        """14.1 IPA 테이블에서 로그를 조회해야 함."""
        # Given
        mock_oracle_adapter = MagicMock()
        mock_oracle_adapter.execute_query.return_value = [
            {
                "SQL_ID": "sql_001",
                "SQL_TEXT": "SELECT * FROM users WHERE id = 1",
                "EXEC_COUNT": 100,
                "ERROR_COUNT": 0,
                "COLLECTED_AT": datetime(2026, 1, 8, 10, 0, 0),
                "SCHEMA_NAME": "APP",
            },
            {
                "SQL_ID": "sql_002",
                "SQL_TEXT": "SELECT * FROM orders WHERE user_id = 2",
                "EXEC_COUNT": 50,
                "ERROR_COUNT": 0,
                "COLLECTED_AT": datetime(2026, 1, 8, 10, 0, 0),
                "SCHEMA_NAME": "APP",
            },
        ]

        collector = LogCollector(oracle_adapter=mock_oracle_adapter)

        # When
        logs = collector.collect()

        # Then
        mock_oracle_adapter.execute_query.assert_called_once()
        assert len(logs) == 2
        assert isinstance(logs[0], RawSQLLog)
        assert logs[0].sql_id == "sql_001"
        assert logs[0].sql_text == "SELECT * FROM users WHERE id = 1"
        assert logs[0].exec_count == 100
        assert logs[0].error_count == 0
        assert logs[1].sql_id == "sql_002"

    def test_collect_with_date_range_should_filter_by_date(self) -> None:
        """14.2 날짜 범위로 필터링해야 함."""
        # Given
        mock_oracle_adapter = MagicMock()
        mock_oracle_adapter.execute_query.return_value = [
            {
                "SQL_ID": "sql_001",
                "SQL_TEXT": "SELECT * FROM users",
                "EXEC_COUNT": 100,
                "ERROR_COUNT": 0,
                "COLLECTED_AT": datetime(2026, 1, 8, 10, 0, 0),
                "SCHEMA_NAME": "APP",
            },
        ]

        collector = LogCollector(oracle_adapter=mock_oracle_adapter)

        start_date = datetime(2026, 1, 1)
        end_date = datetime(2026, 1, 31)

        # When
        logs = collector.collect(start_date=start_date, end_date=end_date)

        # Then
        call_args = mock_oracle_adapter.execute_query.call_args
        query = call_args[0][0]

        # 쿼리에 날짜 조건이 포함되어야 함
        assert "COLLECTED_AT >=" in query or "COLLECTED_AT BETWEEN" in query
        assert len(logs) == 1

    def test_collect_with_limit_should_apply_rownum(self) -> None:
        """14.3 limit 적용 시 ROWNUM 제한을 사용해야 함."""
        # Given
        mock_oracle_adapter = MagicMock()
        mock_oracle_adapter.execute_query.return_value = [
            {
                "SQL_ID": "sql_001",
                "SQL_TEXT": "SELECT * FROM users",
                "EXEC_COUNT": 100,
                "ERROR_COUNT": 0,
                "COLLECTED_AT": datetime(2026, 1, 8, 10, 0, 0),
                "SCHEMA_NAME": "APP",
            },
        ]

        collector = LogCollector(oracle_adapter=mock_oracle_adapter)

        # When
        logs = collector.collect(limit=100)

        # Then
        call_args = mock_oracle_adapter.execute_query.call_args
        query = call_args[0][0]

        # 쿼리에 ROWNUM 제한이 포함되어야 함
        assert "ROWNUM <= 100" in query
        assert len(logs) == 1

