"""Core 데이터 모델 정의."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class SQLOperation(Enum):
    """SQL 연산 타입."""

    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    ALTER = "ALTER"
    DROP = "DROP"


@dataclass
class RawSQLLog:
    """IPA 로그에서 수집한 원본 SQL 로그."""

    sql_id: str
    sql_text: str
    exec_count: int
    error_count: int
    collected_at: datetime
    schema_name: Optional[str] = None


@dataclass
class NormalizedSQL:
    """정규화된 SQL."""

    original_sql_id: str
    normalized_text: str
    tables: list[str]
    columns: list[str]
    template_hash: Optional[str] = None


@dataclass
class SQLTemplate:
    """SQL 템플릿 - 벡터 저장소에 인덱싱되는 최종 형태."""

    template_id: str
    template_text: str
    description: str
    tables: list[str]
    columns: list[str]
    embedding: Optional[list[float]] = None
    exec_count: int = 0


@dataclass
class TableMetadata:
    """테이블 메타데이터."""

    table_name: str
    owner: str
    tablespace_name: Optional[str] = None
    num_rows: Optional[int] = None
    comments: Optional[str] = None


@dataclass
class ColumnMetadata:
    """컬럼 메타데이터."""

    column_name: str
    data_type: str
    data_length: int
    nullable: bool
    column_id: int
    comments: Optional[str] = None


@dataclass
class GlossaryTerm:
    """용어 사전 항목."""

    term: str
    korean_name: str
    description: str
    category: Optional[str] = None

