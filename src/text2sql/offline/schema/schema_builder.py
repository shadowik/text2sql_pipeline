"""스키마 빌더 - Oracle에서 스키마 메타데이터를 조회."""

from typing import Any, Optional

from text2sql.core.models import ColumnMetadata, TableMetadata


class SchemaBuilder:
    """스키마 메타데이터를 빌드하는 서비스."""

    def __init__(self, oracle_adapter: Any) -> None:
        """스키마 빌더 초기화.

        Args:
            oracle_adapter: Oracle 어댑터
        """
        self._oracle_adapter = oracle_adapter

    def get_table_metadata(self, table_name: str) -> Optional[TableMetadata]:
        """테이블 메타데이터를 조회.

        Args:
            table_name: 테이블 이름

        Returns:
            TableMetadata 객체 또는 None
        """
        query = f"""
            SELECT TABLE_NAME, OWNER, TABLESPACE_NAME, NUM_ROWS
            FROM ALL_TABLES
            WHERE TABLE_NAME = '{table_name}'
        """
        result = self._oracle_adapter.execute_query(query)

        if not result:
            return None

        row = result[0]
        return TableMetadata(
            table_name=row["TABLE_NAME"],
            owner=row["OWNER"],
            tablespace_name=row.get("TABLESPACE_NAME"),
            num_rows=row.get("NUM_ROWS"),
        )

    def get_column_metadata(self, table_name: str) -> list[ColumnMetadata]:
        """컬럼 메타데이터를 조회.

        Args:
            table_name: 테이블 이름

        Returns:
            ColumnMetadata 객체 리스트
        """
        query = f"""
            SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, NULLABLE, COLUMN_ID
            FROM ALL_TAB_COLUMNS
            WHERE TABLE_NAME = '{table_name}'
            ORDER BY COLUMN_ID
        """
        result = self._oracle_adapter.execute_query(query)

        return [
            ColumnMetadata(
                column_name=row["COLUMN_NAME"],
                data_type=row["DATA_TYPE"],
                data_length=row["DATA_LENGTH"],
                nullable=row["NULLABLE"] == "Y",
                column_id=row["COLUMN_ID"],
            )
            for row in result
        ]

    def get_table_comments(self, table_name: str) -> Optional[str]:
        """테이블 코멘트를 조회.

        Args:
            table_name: 테이블 이름

        Returns:
            테이블 코멘트 문자열 또는 None
        """
        query = f"""
            SELECT TABLE_NAME, COMMENTS
            FROM ALL_TAB_COMMENTS
            WHERE TABLE_NAME = '{table_name}'
        """
        result = self._oracle_adapter.execute_query(query)

        if not result:
            return None

        return result[0].get("COMMENTS")

    def get_column_comments(self, table_name: str) -> dict[str, Optional[str]]:
        """컬럼 코멘트들을 조회.

        Args:
            table_name: 테이블 이름

        Returns:
            컬럼명 -> 코멘트 딕셔너리
        """
        query = f"""
            SELECT COLUMN_NAME, COMMENTS
            FROM ALL_COL_COMMENTS
            WHERE TABLE_NAME = '{table_name}'
        """
        result = self._oracle_adapter.execute_query(query)

        return {row["COLUMN_NAME"]: row.get("COMMENTS") for row in result}

