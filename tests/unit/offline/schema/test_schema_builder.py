"""SchemaBuilder 테스트."""

from unittest.mock import MagicMock

import pytest

from text2sql.offline.schema.schema_builder import SchemaBuilder


class TestSchemaBuilder:
    """SchemaBuilder 테스트 클래스."""

    def test_get_table_metadata_returns_table_info(self) -> None:
        """16.1 Oracle에서 테이블 메타데이터 조회 테스트."""
        # Given
        mock_oracle_adapter = MagicMock()
        mock_oracle_adapter.execute_query.return_value = [
            {
                "TABLE_NAME": "USERS",
                "OWNER": "APP_OWNER",
                "TABLESPACE_NAME": "DATA_TS",
                "NUM_ROWS": 10000,
            }
        ]
        schema_builder = SchemaBuilder(mock_oracle_adapter)

        # When
        result = schema_builder.get_table_metadata("USERS")

        # Then
        assert result is not None
        assert result.table_name == "USERS"
        assert result.owner == "APP_OWNER"
        assert result.tablespace_name == "DATA_TS"
        assert result.num_rows == 10000

    def test_get_column_metadata_returns_column_list(self) -> None:
        """16.2 Oracle에서 컬럼 메타데이터 조회 테스트."""
        # Given
        mock_oracle_adapter = MagicMock()
        mock_oracle_adapter.execute_query.return_value = [
            {
                "COLUMN_NAME": "USER_ID",
                "DATA_TYPE": "NUMBER",
                "DATA_LENGTH": 22,
                "NULLABLE": "N",
                "COLUMN_ID": 1,
            },
            {
                "COLUMN_NAME": "USER_NAME",
                "DATA_TYPE": "VARCHAR2",
                "DATA_LENGTH": 100,
                "NULLABLE": "Y",
                "COLUMN_ID": 2,
            },
        ]
        schema_builder = SchemaBuilder(mock_oracle_adapter)

        # When
        result = schema_builder.get_column_metadata("USERS")

        # Then
        assert len(result) == 2
        assert result[0].column_name == "USER_ID"
        assert result[0].data_type == "NUMBER"
        assert result[0].nullable is False
        assert result[1].column_name == "USER_NAME"
        assert result[1].data_type == "VARCHAR2"
        assert result[1].nullable is True

    def test_get_table_comments_returns_comment(self) -> None:
        """16.3 테이블 코멘트 추출 테스트."""
        # Given
        mock_oracle_adapter = MagicMock()
        mock_oracle_adapter.execute_query.return_value = [
            {
                "TABLE_NAME": "USERS",
                "COMMENTS": "사용자 정보 테이블",
            }
        ]
        schema_builder = SchemaBuilder(mock_oracle_adapter)

        # When
        result = schema_builder.get_table_comments("USERS")

        # Then
        assert result == "사용자 정보 테이블"

    def test_get_column_comments_returns_comments_dict(self) -> None:
        """16.4 컬럼 코멘트 추출 테스트."""
        # Given
        mock_oracle_adapter = MagicMock()
        mock_oracle_adapter.execute_query.return_value = [
            {
                "COLUMN_NAME": "USER_ID",
                "COMMENTS": "사용자 고유 식별자",
            },
            {
                "COLUMN_NAME": "USER_NAME",
                "COMMENTS": "사용자 이름",
            },
            {
                "COLUMN_NAME": "EMAIL",
                "COMMENTS": None,
            },
        ]
        schema_builder = SchemaBuilder(mock_oracle_adapter)

        # When
        result = schema_builder.get_column_comments("USERS")

        # Then
        assert len(result) == 3
        assert result["USER_ID"] == "사용자 고유 식별자"
        assert result["USER_NAME"] == "사용자 이름"
        assert result["EMAIL"] is None

