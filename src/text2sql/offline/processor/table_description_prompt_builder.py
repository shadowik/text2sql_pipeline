"""테이블 설명 생성을 위한 LLM 프롬프트 빌더.

설계서 4.6절에 따라 불충분한 테이블/컬럼 코멘트를 LLM을 통해 보강하기 위한
프롬프트를 생성합니다.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TableRegistry:
    """테이블 레지스트리 (설계서 4.4절 스키마)."""

    table_name: str
    schema_name: str
    system_comment: Optional[str]  # 원본 DB 코멘트 (누락/불충분 가능)
    sql_count: int
    first_seen_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    meta_status: str = "pending"
    description_source: Optional[str] = None


@dataclass
class ColumnRegistry:
    """컬럼 레지스트리 (설계서 4.5절 스키마)."""

    column_name: str
    data_type: str
    system_comment: Optional[str]  # 원본 DB 코멘트 (누락/불충분 가능)
    sample_values: list[Any]
    statistics: dict[str, Any]
    usage_context: dict[str, int]


@dataclass
class JoinPattern:
    """조인 패턴."""

    target_table: str
    join_column: str
    join_count: int


@dataclass
class WherePattern:
    """WHERE 조건 패턴."""

    condition: str
    count: int


class TableDescriptionPromptBuilder:
    """테이블 설명 생성을 위한 LLM 프롬프트 빌더.

    설계서 4.6.2절의 프롬프트 형식을 따릅니다.
    """

    def __init__(self, language: str = "ko") -> None:
        """빌더 초기화.

        Args:
            language: 출력 언어 (ko: 한국어, en: 영어)
        """
        self._language = language

    def build_table_description_prompt(
        self,
        table_registry: TableRegistry,
        column_registries: list[ColumnRegistry],
        sample_sqls: list[str],
        join_patterns: list[JoinPattern],
        where_patterns: list[WherePattern],
        sample_data: list[dict[str, Any]],
    ) -> str:
        """테이블 설명 생성을 위한 LLM 프롬프트를 구성합니다.

        Args:
            table_registry: 테이블 레지스트리 정보
            column_registries: 컬럼 레지스트리 목록
            sample_sqls: 해당 테이블이 사용된 대표 SQL 목록
            join_patterns: 자주 조인되는 테이블 패턴
            where_patterns: 자주 사용되는 WHERE 조건 패턴
            sample_data: 샘플 레코드 데이터

        Returns:
            LLM에 전달할 프롬프트 문자열
        """
        # JSON 형태의 입력 데이터 구성 (설계서 4.6.2절 형식)
        input_data = self._build_input_data(
            table_registry,
            column_registries,
            sample_sqls,
            join_patterns,
            where_patterns,
            sample_data,
        )

        # 시스템 프롬프트와 사용자 프롬프트 구성
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(input_data)

        return f"{system_prompt}\n\n{user_prompt}"

    def build_column_description_prompt(
        self,
        table_name: str,
        column_registry: ColumnRegistry,
        sample_sqls: list[str],
    ) -> str:
        """컬럼 설명 생성을 위한 LLM 프롬프트를 구성합니다.

        Args:
            table_name: 테이블명
            column_registry: 컬럼 레지스트리 정보
            sample_sqls: 해당 컬럼이 사용된 대표 SQL 목록

        Returns:
            LLM에 전달할 프롬프트 문자열
        """
        input_data = {
            "task": "컬럼 설명 생성",
            "table_name": table_name,
            "column_name": column_registry.column_name,
            "data_type": column_registry.data_type,
            "system_comment": column_registry.system_comment,
            "sample_values": column_registry.sample_values[:10],  # 최대 10개
            "statistics": column_registry.statistics,
            "usage_context": column_registry.usage_context,
            "sample_sqls": sample_sqls[:5],  # 최대 5개
        }

        system_prompt = self._build_column_system_prompt()
        user_prompt = f"### 입력 데이터\n```json\n{json.dumps(input_data, ensure_ascii=False, indent=2)}\n```"

        return f"{system_prompt}\n\n{user_prompt}"

    def _build_input_data(
        self,
        table_registry: TableRegistry,
        column_registries: list[ColumnRegistry],
        sample_sqls: list[str],
        join_patterns: list[JoinPattern],
        where_patterns: list[WherePattern],
        sample_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """입력 데이터를 JSON 형태로 구성합니다."""
        columns_info = []
        for col in column_registries:
            col_info = {
                "name": col.column_name,
                "type": col.data_type,
                "system_comment": col.system_comment,
                "sample_values": col.sample_values[:5],  # 최대 5개
            }
            if col.statistics:
                col_info["statistics"] = col.statistics
            columns_info.append(col_info)

        join_patterns_formatted = [
            f"{jp.target_table} ({jp.join_column})" for jp in join_patterns
        ]

        where_patterns_formatted = [
            {"condition": wp.condition, "frequency": wp.count} for wp in where_patterns
        ]

        return {
            "task": "테이블 설명 생성",
            "table_name": table_registry.table_name,
            "schema_name": table_registry.schema_name,
            "system_comment": table_registry.system_comment,
            "sql_count": table_registry.sql_count,
            "columns": columns_info,
            "sample_sqls": sample_sqls[:5],  # 최대 5개
            "join_patterns": join_patterns_formatted,
            "where_patterns": where_patterns_formatted,
            "sample_data": sample_data[:10],  # 최대 10개
        }

    def _build_system_prompt(self) -> str:
        """테이블 설명 생성용 시스템 프롬프트를 구성합니다."""
        if self._language == "ko":
            return """### 시스템 프롬프트

당신은 MES(Manufacturing Execution System) 데이터베이스 전문가입니다.
제공된 정보를 분석하여 테이블에 대한 상세한 설명을 생성해주세요.

### 분석 시 고려사항
1. **테이블 목적**: SQL 사용 패턴을 분석하여 테이블의 주요 사용 목적을 파악합니다.
2. **비즈니스 컨텍스트**: 컬럼명, 샘플 데이터, WHERE 조건을 통해 비즈니스 의미를 추론합니다.
3. **테이블 관계**: JOIN 패턴을 통해 다른 테이블과의 관계를 파악합니다.
4. **도메인 지식**: MES 도메인 (장비, 로트, 공정, 수율, 결함 등)의 용어와 개념을 활용합니다.

### 출력 형식 (JSON)
```json
{
  "table_description": "테이블의 목적과 역할을 상세히 기술 (2-3문장)",
  "primary_use_cases": ["주요 사용 사례 1", "주요 사용 사례 2", "주요 사용 사례 3"],
  "related_domain": "관련 도메인 (예: MES/설비관리, MES/품질관리, MES/생산이력 등)",
  "column_descriptions": {
    "컬럼명1": "컬럼 설명",
    "컬럼명2": "컬럼 설명"
  },
  "business_rules": ["발견된 비즈니스 규칙 1", "발견된 비즈니스 규칙 2"]
}
```

### 중요 지침
- 원본 DB의 system_comment가 null이거나 불충분한 경우, SQL 사용 패턴과 샘플 데이터를 기반으로 설명을 생성합니다.
- 추측이 아닌 제공된 데이터에 근거한 설명을 작성합니다.
- 한국어로 작성해주세요."""
        else:
            return """### System Prompt

You are an expert in MES (Manufacturing Execution System) databases.
Analyze the provided information and generate detailed descriptions for the table.

### Analysis Considerations
1. **Table Purpose**: Analyze SQL usage patterns to understand the main purpose.
2. **Business Context**: Infer business meaning from column names, sample data, and WHERE conditions.
3. **Table Relationships**: Understand relationships with other tables through JOIN patterns.
4. **Domain Knowledge**: Utilize MES domain terminology (equipment, lot, process, yield, defect, etc.).

### Output Format (JSON)
```json
{
  "table_description": "Detailed description of table purpose and role (2-3 sentences)",
  "primary_use_cases": ["Use case 1", "Use case 2", "Use case 3"],
  "related_domain": "Related domain (e.g., MES/Equipment Management, MES/Quality Control)",
  "column_descriptions": {
    "column1": "column description",
    "column2": "column description"
  },
  "business_rules": ["Business rule 1", "Business rule 2"]
}
```

### Important Guidelines
- If system_comment is null or insufficient, generate descriptions based on SQL patterns and sample data.
- Write descriptions based on provided data, not speculation.
- Write in English."""

    def _build_column_system_prompt(self) -> str:
        """컬럼 설명 생성용 시스템 프롬프트를 구성합니다."""
        if self._language == "ko":
            return """### 시스템 프롬프트

당신은 MES 데이터베이스 전문가입니다.
제공된 컬럼 정보를 분석하여 컬럼에 대한 상세한 설명을 생성해주세요.

### 분석 시 고려사항
1. **데이터 타입**: NUMBER, VARCHAR, DATE 등 타입에 따른 특성을 고려합니다.
2. **샘플 값**: 실제 데이터 패턴을 분석하여 의미를 파악합니다.
3. **사용 컨텍스트**: SELECT/WHERE/JOIN/GROUP BY 사용 빈도를 통해 중요도를 판단합니다.
4. **통계 정보**: min/max/avg 등의 통계로 데이터 범위를 파악합니다.

### 출력 형식 (JSON)
```json
{
  "column_description": "컬럼의 목적과 저장되는 데이터에 대한 설명 (1-2문장)",
  "data_characteristics": "데이터 특성 (코드값, 식별자, 측정값 등)",
  "business_meaning": "비즈니스 관점에서의 의미",
  "value_examples": "대표적인 값들의 의미 설명"
}
```

### 중요 지침
- 원본 DB의 system_comment가 null이거나 불충분한 경우, 샘플 값과 통계를 기반으로 설명을 생성합니다.
- 한국어로 작성해주세요."""
        else:
            return """### System Prompt

You are an MES database expert.
Analyze the provided column information and generate detailed descriptions.

### Output Format (JSON)
```json
{
  "column_description": "Description of column purpose and stored data (1-2 sentences)",
  "data_characteristics": "Data characteristics (code values, identifiers, measurements, etc.)",
  "business_meaning": "Business perspective meaning",
  "value_examples": "Explanation of representative values"
}
```"""

    def _build_user_prompt(self, input_data: dict[str, Any]) -> str:
        """사용자 프롬프트를 구성합니다."""
        return f"### 입력 데이터\n```json\n{json.dumps(input_data, ensure_ascii=False, indent=2)}\n```"

    def is_comment_insufficient(
        self, system_comment: Optional[str], min_length: int = 5
    ) -> bool:
        """시스템 코멘트가 불충분한지 확인합니다.

        Args:
            system_comment: 원본 DB의 코멘트
            min_length: 충분한 코멘트로 간주할 최소 길이

        Returns:
            불충분하면 True, 충분하면 False
        """
        if system_comment is None:
            return True
        if len(system_comment.strip()) < min_length:
            return True
        # 약어만 있는 경우 (모두 대문자이고 짧은 경우)
        if system_comment.isupper() and len(system_comment) <= 10:
            return True
        return False


def load_sample_data(
    base_path: str,
) -> tuple[
    list[TableRegistry],
    dict[str, list[ColumnRegistry]],
    dict[str, dict],
    dict[str, list[dict]],
]:
    """샘플 데이터 파일들을 로드합니다.

    Args:
        base_path: 샘플 데이터 폴더 경로

    Returns:
        (table_registries, column_registries_by_table, sql_patterns_by_table, sample_data_by_table)
    """
    import os

    # 테이블 레지스트리 로드
    with open(
        os.path.join(base_path, "table_registry_samples.json"), encoding="utf-8"
    ) as f:
        table_data = json.load(f)
    table_registries = [
        TableRegistry(
            table_name=t["table_name"],
            schema_name=t["schema_name"],
            system_comment=t.get("system_comment"),
            sql_count=t["sql_count"],
            first_seen_at=t.get("first_seen_at"),
            last_seen_at=t.get("last_seen_at"),
            meta_status=t.get("meta_status", "pending"),
            description_source=t.get("description_source"),
        )
        for t in table_data
    ]

    # 컬럼 레지스트리 로드
    with open(
        os.path.join(base_path, "column_registry_samples.json"), encoding="utf-8"
    ) as f:
        column_data = json.load(f)
    column_registries_by_table: dict[str, list[ColumnRegistry]] = {}
    for table_name, columns in column_data.items():
        column_registries_by_table[table_name] = [
            ColumnRegistry(
                column_name=c["column_name"],
                data_type=c["data_type"],
                system_comment=c.get("system_comment"),
                sample_values=c.get("sample_values", []),
                statistics=c.get("statistics", {}),
                usage_context=c.get("usage_context", {}),
            )
            for c in columns
        ]

    # SQL 패턴 로드
    with open(
        os.path.join(base_path, "sample_sqls_for_tables.json"), encoding="utf-8"
    ) as f:
        sql_patterns_by_table = json.load(f)

    # 샘플 데이터 로드
    with open(
        os.path.join(base_path, "sample_table_data.json"), encoding="utf-8"
    ) as f:
        sample_data_by_table = json.load(f)

    return (
        table_registries,
        column_registries_by_table,
        sql_patterns_by_table,
        sample_data_by_table,
    )
