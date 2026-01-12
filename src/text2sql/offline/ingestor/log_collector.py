"""로그 수집기."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from text2sql.core.models import RawSQLLog


class JsonLogCollector:
    """JSON 파일에서 SQL 로그를 수집하는 서비스."""

    def __init__(self, json_path: str | Path, limit: int | None = None) -> None:
        """수집기 초기화.

        Args:
            json_path: JSON 파일 경로
            limit: 기본 최대 레코드 수 (None이면 제한 없음)
        """
        self._json_path = Path(json_path)
        self._default_limit = limit

    def collect(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int | None = None,
    ) -> list[RawSQLLog]:
        """JSON 파일에서 로그를 수집.

        Args:
            start_date: 시작 날짜 (None이면 제한 없음)
            end_date: 종료 날짜 (None이면 제한 없음)
            limit: 최대 레코드 수 (None이면 기본값 사용)

        Returns:
            수집된 SQL 로그 리스트
        """
        with open(self._json_path, encoding="utf-8") as f:
            data = json.load(f)

        logs = []
        for item in data:
            collected_at = datetime.fromisoformat(item["collected_at"])

            # 날짜 필터링
            if start_date and collected_at < start_date:
                continue
            if end_date and collected_at > end_date:
                continue

            log = RawSQLLog(
                sql_id=item["sql_id"],
                sql_text=item["sql_text"],
                exec_count=item["exec_count"],
                error_count=item["error_count"],
                collected_at=collected_at,
                schema_name=item.get("schema_name"),
            )
            logs.append(log)

        # exec_count 기준 내림차순 정렬
        logs.sort(key=lambda x: x.exec_count, reverse=True)

        # 리밋 적용
        effective_limit = limit if limit is not None else self._default_limit
        if effective_limit:
            logs = logs[:effective_limit]

        return logs


class LogCollector:
    """IPA 테이블에서 SQL 로그를 수집하는 서비스."""

    DEFAULT_QUERY = """
        SELECT SQL_ID, SQL_TEXT, EXEC_COUNT, ERROR_COUNT, COLLECTED_AT, SCHEMA_NAME
        FROM IPA_SQL_LOG
        ORDER BY EXEC_COUNT DESC
    """

    def __init__(self, oracle_adapter: Any, limit: int | None = None) -> None:
        """수집기 초기화.

        Args:
            oracle_adapter: Oracle 어댑터
            limit: 기본 최대 레코드 수 (None이면 제한 없음)
        """
        self._oracle_adapter = oracle_adapter
        self._default_limit = limit

    def collect(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int | None = None,
    ) -> list[RawSQLLog]:
        """로그를 수집.

        Args:
            start_date: 시작 날짜 (None이면 제한 없음)
            end_date: 종료 날짜 (None이면 제한 없음)
            limit: 최대 레코드 수 (None이면 기본값 사용)

        Returns:
            수집된 SQL 로그 리스트
        """
        effective_limit = limit if limit is not None else self._default_limit
        query = self._build_query(start_date, end_date, effective_limit)
        rows = self._oracle_adapter.execute_query(query)

        return [
            RawSQLLog(
                sql_id=row["SQL_ID"],
                sql_text=row["SQL_TEXT"],
                exec_count=row["EXEC_COUNT"],
                error_count=row["ERROR_COUNT"],
                collected_at=row["COLLECTED_AT"],
                schema_name=row.get("SCHEMA_NAME"),
            )
            for row in rows
        ]

    def _build_query(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
        limit: int | None,
    ) -> str:
        """쿼리를 동적으로 빌드.

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            limit: 최대 레코드 수

        Returns:
            빌드된 SQL 쿼리
        """
        base_query = """
            SELECT SQL_ID, SQL_TEXT, EXEC_COUNT, ERROR_COUNT, COLLECTED_AT, SCHEMA_NAME
            FROM IPA_SQL_LOG
        """

        conditions = []
        if start_date:
            conditions.append(
                f"COLLECTED_AT >= TO_DATE('{start_date.strftime('%Y-%m-%d')}', 'YYYY-MM-DD')"
            )
        if end_date:
            conditions.append(
                f"COLLECTED_AT <= TO_DATE('{end_date.strftime('%Y-%m-%d')}', 'YYYY-MM-DD')"
            )

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " ORDER BY EXEC_COUNT DESC"

        if limit:
            base_query = f"SELECT * FROM ({base_query}) WHERE ROWNUM <= {limit}"

        return base_query

