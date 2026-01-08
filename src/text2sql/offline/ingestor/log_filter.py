"""SQL 로그 필터링 모듈."""

import re
from typing import Optional

from text2sql.core.models import RawSQLLog

# DDL 키워드 패턴 (대소문자 무시)
DDL_PATTERN = re.compile(r"^\s*(CREATE|ALTER|DROP)\s+", re.IGNORECASE)

# DML 키워드 패턴 (대소문자 무시)
DML_PATTERN = re.compile(r"^\s*(INSERT|UPDATE|DELETE)\s+", re.IGNORECASE)

# 배치성/시스템 테이블 패턴 (대소문자 무시)
# SYS. 스키마, DBA_* 테이블, ALL_* 테이블, V$ 뷰 등
# USER_ 다음에 반드시 추가 문자가 와야 함 (USER_TABLES vs users)
BATCH_PATTERN = re.compile(
    r"\b(SYS\.\w+|DBA_\w+|ALL_\w+|USER_\w+|V\$\w+|GV\$\w+)\b", re.IGNORECASE
)

# 집계 함수 패턴 (대소문자 무시)
AGGREGATE_PATTERN = re.compile(
    r"\b(COUNT|SUM|AVG|MAX|MIN|STDDEV|VARIANCE)\s*\(", re.IGNORECASE
)

# 윈도우 함수 패턴 (대소문자 무시)
WINDOW_PATTERN = re.compile(
    r"\b(ROW_NUMBER|RANK|DENSE_RANK|NTILE|LAG|LEAD|FIRST_VALUE|LAST_VALUE)\s*\(",
    re.IGNORECASE,
)


class LogFilter:
    """SQL 로그를 필터링하는 클래스."""

    def filter(
        self, logs: list[RawSQLLog], top_n: Optional[int] = None
    ) -> list[RawSQLLog]:
        """로그 목록을 필터링한다.
        
        Args:
            logs: 필터링할 RawSQLLog 목록
            top_n: 반환할 최대 로그 수 (exec_count 기준 정렬)
            
        Returns:
            필터링된 RawSQLLog 목록
        """
        filtered = [log for log in logs if self._is_valid(log)]
        
        # exec_count 기준 내림차순, 같으면 집계 함수 포함 여부로 정렬
        sorted_logs = sorted(
            filtered,
            key=lambda x: (x.exec_count, self._has_analytics(x.sql_text)),
            reverse=True,
        )
        
        if top_n is None:
            return sorted_logs
        return sorted_logs[:top_n]

    def _is_valid(self, log: RawSQLLog) -> bool:
        """로그가 유효한지 확인한다."""
        if log.error_count > 0:
            return False
        if self._is_ddl(log.sql_text):
            return False
        if self._is_dml(log.sql_text):
            return False
        if self._is_batch_query(log.sql_text):
            return False
        return True

    def _is_ddl(self, sql_text: str) -> bool:
        """DDL 쿼리인지 확인한다."""
        return bool(DDL_PATTERN.match(sql_text))

    def _is_dml(self, sql_text: str) -> bool:
        """DML 쿼리인지 확인한다."""
        return bool(DML_PATTERN.match(sql_text))

    def _is_batch_query(self, sql_text: str) -> bool:
        """배치성/시스템 쿼리인지 확인한다."""
        return bool(BATCH_PATTERN.search(sql_text))

    def _has_analytics(self, sql_text: str) -> bool:
        """집계 함수 또는 윈도우 함수가 포함되어 있는지 확인한다."""
        return bool(
            AGGREGATE_PATTERN.search(sql_text) or WINDOW_PATTERN.search(sql_text)
        )

