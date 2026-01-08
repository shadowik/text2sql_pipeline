"""LogFilter 유닛 테스트."""

from datetime import datetime

import pytest

from text2sql.core.models import RawSQLLog
from text2sql.offline.ingestor.log_filter import LogFilter


class TestLogFilterExcludeErrors:
    """5.1 error_count > 0인 로그 제외 테스트."""

    def test_excludes_logs_with_error_count_greater_than_zero(self):
        """error_count가 0보다 큰 로그는 제외해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM users",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
            RawSQLLog(
                sql_id="sql2",
                sql_text="SELECT * FROM orders",
                exec_count=50,
                error_count=5,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 1
        assert result[0].sql_id == "sql1"

    def test_includes_logs_with_zero_error_count(self):
        """error_count가 0인 로그는 포함해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM users",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
            RawSQLLog(
                sql_id="sql2",
                sql_text="SELECT * FROM products",
                exec_count=200,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 2

    def test_returns_empty_when_all_logs_have_errors(self):
        """모든 로그에 에러가 있으면 빈 리스트를 반환해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM users",
                exec_count=100,
                error_count=1,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 0


class TestLogFilterExcludeDDL:
    """5.2 DDL 쿼리 제외 테스트 (CREATE, ALTER, DROP)."""

    def test_excludes_create_table_query(self):
        """CREATE TABLE 쿼리는 제외해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="CREATE TABLE users (id NUMBER)",
                exec_count=1,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 0

    def test_excludes_alter_table_query(self):
        """ALTER TABLE 쿼리는 제외해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="ALTER TABLE users ADD COLUMN name VARCHAR2(100)",
                exec_count=1,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 0

    def test_excludes_drop_table_query(self):
        """DROP TABLE 쿼리는 제외해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="DROP TABLE users",
                exec_count=1,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 0

    def test_excludes_ddl_case_insensitive(self):
        """DDL 키워드는 대소문자를 구분하지 않아야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="create table users (id NUMBER)",
                exec_count=1,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 0


class TestLogFilterExcludeDML:
    """5.3 DML 쿼리 제외 테스트 (INSERT, UPDATE, DELETE)."""

    def test_excludes_insert_query(self):
        """INSERT 쿼리는 제외해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="INSERT INTO users (id, name) VALUES (1, 'test')",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 0

    def test_excludes_update_query(self):
        """UPDATE 쿼리는 제외해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="UPDATE users SET name = 'test' WHERE id = 1",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 0

    def test_excludes_delete_query(self):
        """DELETE 쿼리는 제외해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="DELETE FROM users WHERE id = 1",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 0

    def test_excludes_dml_case_insensitive(self):
        """DML 키워드는 대소문자를 구분하지 않아야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="insert into users (id) values (1)",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 0


class TestLogFilterSelectOnly:
    """5.4 SELECT 쿼리만 통과 테스트."""

    def test_includes_select_query(self):
        """SELECT 쿼리는 통과해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM users WHERE id = 1",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 1
        assert result[0].sql_id == "sql1"

    def test_includes_select_with_subquery(self):
        """서브쿼리를 포함한 SELECT도 통과해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM employees WHERE id IN (SELECT emp_id FROM orders)",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 1

    def test_includes_select_case_insensitive(self):
        """SELECT 키워드는 대소문자를 구분하지 않아야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="select * from users",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 1

    def test_filters_mixed_queries_returns_only_select(self):
        """SELECT, DML, DDL이 섞여있을 때 SELECT만 반환해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM users",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
            RawSQLLog(
                sql_id="sql2",
                sql_text="INSERT INTO users (id) VALUES (1)",
                exec_count=50,
                error_count=0,
                collected_at=datetime.now(),
            ),
            RawSQLLog(
                sql_id="sql3",
                sql_text="CREATE TABLE temp (id NUMBER)",
                exec_count=10,
                error_count=0,
                collected_at=datetime.now(),
            ),
            RawSQLLog(
                sql_id="sql4",
                sql_text="SELECT name FROM products",
                exec_count=200,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 2
        # 결과는 exec_count 기준 내림차순으로 정렬됨
        result_ids = [r.sql_id for r in result]
        assert "sql1" in result_ids
        assert "sql4" in result_ids


class TestLogFilterTopN:
    """5.5 exec_count 기준 TOP-N 우선순위 선정 테스트."""

    def test_returns_top_n_by_exec_count(self):
        """exec_count가 높은 순서로 TOP-N을 반환해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM users",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
            RawSQLLog(
                sql_id="sql2",
                sql_text="SELECT * FROM orders",
                exec_count=500,
                error_count=0,
                collected_at=datetime.now(),
            ),
            RawSQLLog(
                sql_id="sql3",
                sql_text="SELECT * FROM products",
                exec_count=200,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs, top_n=2)

        assert len(result) == 2
        assert result[0].sql_id == "sql2"  # exec_count: 500
        assert result[1].sql_id == "sql3"  # exec_count: 200

    def test_returns_all_when_top_n_is_none(self):
        """top_n이 None이면 모든 결과를 반환해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM users",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
            RawSQLLog(
                sql_id="sql2",
                sql_text="SELECT * FROM orders",
                exec_count=500,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs, top_n=None)

        assert len(result) == 2

    def test_returns_all_when_top_n_exceeds_count(self):
        """top_n이 로그 수보다 크면 모든 결과를 반환해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM users",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs, top_n=10)

        assert len(result) == 1


class TestLogFilterExcludeBatchPatterns:
    """5.6 배치성 패턴 제외 테스트."""

    def test_excludes_sys_schema_query(self):
        """SYS. 스키마 쿼리는 제외해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM SYS.DBA_TABLES",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 0

    def test_excludes_dba_prefix_tables(self):
        """DBA_ 접두사 테이블 쿼리는 제외해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM DBA_OBJECTS WHERE object_type = 'TABLE'",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 0

    def test_excludes_v_dollar_views(self):
        """V$ 뷰 쿼리는 제외해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM V$SESSION WHERE status = 'ACTIVE'",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 0

    def test_excludes_all_prefix_tables(self):
        """ALL_ 접두사 테이블 쿼리는 제외해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM ALL_TAB_COLUMNS WHERE table_name = 'USERS'",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 0

    def test_includes_normal_business_query(self):
        """일반 비즈니스 쿼리는 통과해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM MES_PRD_M11 WHERE lot_id = 'LOT001'",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert len(result) == 1


class TestLogFilterPrioritizeAggregation:
    """5.7 집계 함수 포함 쿼리 우선 선정 테스트."""

    def test_prioritizes_queries_with_count(self):
        """COUNT 함수가 포함된 쿼리가 우선되어야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM products WHERE id = 1",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
            RawSQLLog(
                sql_id="sql2",
                sql_text="SELECT COUNT(*) FROM products GROUP BY category",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        # 같은 exec_count일 때 집계 함수가 있는 쿼리가 먼저 와야 함
        assert result[0].sql_id == "sql2"

    def test_prioritizes_queries_with_sum_avg_max_min(self):
        """SUM, AVG, MAX, MIN 함수가 포함된 쿼리가 우선되어야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM sales",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
            RawSQLLog(
                sql_id="sql2",
                sql_text="SELECT SUM(amount), AVG(price) FROM sales",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert result[0].sql_id == "sql2"

    def test_prioritizes_queries_with_window_functions(self):
        """윈도우 함수가 포함된 쿼리가 우선되어야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM employees",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
            RawSQLLog(
                sql_id="sql2",
                sql_text="SELECT name, ROW_NUMBER() OVER (PARTITION BY dept) FROM employees",
                exec_count=100,
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        assert result[0].sql_id == "sql2"

    def test_exec_count_still_primary_sorting_factor(self):
        """exec_count가 높으면 집계 함수 유무보다 우선해야 한다."""
        logs = [
            RawSQLLog(
                sql_id="sql1",
                sql_text="SELECT * FROM products WHERE id = 1",
                exec_count=1000,  # 높은 exec_count
                error_count=0,
                collected_at=datetime.now(),
            ),
            RawSQLLog(
                sql_id="sql2",
                sql_text="SELECT COUNT(*) FROM products",
                exec_count=100,  # 낮은 exec_count
                error_count=0,
                collected_at=datetime.now(),
            ),
        ]

        log_filter = LogFilter()
        result = log_filter.filter(logs)

        # exec_count가 더 높은 쿼리가 먼저
        assert result[0].sql_id == "sql1"
