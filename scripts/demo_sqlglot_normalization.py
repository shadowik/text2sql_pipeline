#!/usr/bin/env python3
"""
sqlglot 기반 SQL 정규화 예제 스크립트.

이 스크립트는 docs/pipeline_design_draft.md의 "4.2 정규화 및 템플릿 해시" 섹션에서
설명하는 sqlglot 파싱을 사용한 SQL 정규화를 시연합니다.

기능:
1. 바인드 변수/리터럴 제거 (WHERE COL = :B1, 날짜 상수 등)
2. 키워드/공백/대소문자 정규화
3. 정규화된 SQL의 해시 생성
"""

import hashlib

import sqlglot
from sqlglot import exp
from sqlglot.optimizer import normalize_identifiers


def normalize_sql(
    sql: str,
    dialect: str = "oracle",
    placeholder: str = ":placeholder",
) -> str:
    """SQL을 정규화하여 canonical form으로 변환한다.

    Args:
        sql: 원본 SQL 문자열
        dialect: SQL 방언 (기본값: oracle)
        placeholder: 리터럴 대체 문자열

    Returns:
        정규화된 SQL 문자열
    """
    try:
        # SQL 파싱
        parsed = sqlglot.parse_one(sql, dialect=dialect)
    except Exception as e:
        print(f"파싱 실패: {e}")
        return sql

    # 리터럴을 placeholder로 치환
    for node in parsed.walk():
        if isinstance(node, exp.Literal):
            # 문자열/숫자 리터럴을 placeholder로 변경
            node.replace(exp.Placeholder())

    # 정규화된 SQL 생성 (대문자 키워드, 일관된 공백)
    normalized = parsed.sql(dialect=dialect, normalize=True, pretty=False)

    return normalized


def normalize_identifiers_case(sql: str, dialect: str = "oracle") -> str:
    """식별자(테이블명, 컬럼명)의 대소문자를 정규화한다.

    Args:
        sql: SQL 문자열
        dialect: SQL 방언

    Returns:
        식별자가 정규화된 SQL
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        normalized = normalize_identifiers.normalize_identifiers(
            parsed, dialect=dialect
        )
        return normalized.sql(dialect=dialect)
    except Exception as e:
        print(f"식별자 정규화 실패: {e}")
        return sql


def generate_template_hash(normalized_sql: str) -> str:
    """정규화된 SQL의 해시를 생성한다.

    Args:
        normalized_sql: 정규화된 SQL 문자열

    Returns:
        SHA-256 해시 문자열
    """
    return hashlib.sha256(normalized_sql.encode()).hexdigest()


def extract_tables(sql: str, dialect: str = "oracle") -> list[str]:
    """SQL에서 테이블명을 추출한다.

    Args:
        sql: SQL 문자열
        dialect: SQL 방언

    Returns:
        테이블명 리스트
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        tables = []

        for table in parsed.find_all(exp.Table):
            table_name = table.name
            if table.db:
                table_name = f"{table.db}.{table_name}"
            tables.append(table_name)

        return list(set(tables))
    except Exception as e:
        print(f"테이블 추출 실패: {e}")
        return []


def extract_columns(sql: str, dialect: str = "oracle") -> list[str]:
    """SQL에서 SELECT 절의 컬럼명을 추출한다.

    Args:
        sql: SQL 문자열
        dialect: SQL 방언

    Returns:
        컬럼명 리스트
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        columns = []

        for select in parsed.find_all(exp.Select):
            for expression in select.expressions:
                if isinstance(expression, exp.Star):
                    columns.append("*")
                elif isinstance(expression, exp.Column):
                    columns.append(expression.name)
                elif isinstance(expression, exp.Alias):
                    # Alias의 경우 별칭 이름 추출
                    columns.append(expression.alias)
                else:
                    # 다른 표현식의 경우 문자열로 변환
                    columns.append(str(expression))

        return columns
    except Exception as e:
        print(f"컬럼 추출 실패: {e}")
        return []


def extract_where_conditions(sql: str, dialect: str = "oracle") -> list[str]:
    """SQL의 WHERE 절 조건을 추출한다.

    Args:
        sql: SQL 문자열
        dialect: SQL 방언

    Returns:
        조건 표현식 리스트
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        conditions = []

        for where in parsed.find_all(exp.Where):
            for cond in where.walk():
                if isinstance(cond, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
                    conditions.append(cond.sql(dialect=dialect))

        return conditions
    except Exception as e:
        print(f"WHERE 조건 추출 실패: {e}")
        return []


def extract_joins(sql: str, dialect: str = "oracle") -> list[dict]:
    """SQL에서 JOIN 정보를 추출한다.

    Args:
        sql: SQL 문자열
        dialect: SQL 방언

    Returns:
        JOIN 정보 딕셔너리 리스트
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        joins = []

        for join in parsed.find_all(exp.Join):
            join_info = {
                "type": join.side if join.side else "INNER",
                "table": join.this.name if isinstance(join.this, exp.Table) else str(join.this),
                "on_condition": join.args.get("on").sql(dialect=dialect) if join.args.get("on") else None
            }
            joins.append(join_info)

        return joins
    except Exception as e:
        print(f"JOIN 추출 실패: {e}")
        return []


def demo():
    """sqlglot 정규화 기능 데모."""
    # 테스트 SQL 예시들
    test_sqls = [
        # 기본 SELECT 쿼리
        """
        SELECT equip_id, mes_stat_typ, created_at
        FROM r3_equip_status
        WHERE area_id = 'ETCH'
          AND mes_stat_typ = 'Down'
          AND created_at > TO_DATE('2025-01-01', 'YYYY-MM-DD')
        """,

        # 바인드 변수가 포함된 쿼리
        """
        SELECT e.equip_id, e.equip_name, s.status_code
        FROM equipment e
        INNER JOIN equip_status s ON e.equip_id = s.equip_id
        WHERE e.area_id = :area
          AND s.check_date BETWEEN :start_date AND :end_date
        ORDER BY e.equip_id
        """,

        # 서브쿼리가 포함된 복잡한 쿼리
        """
        SELECT lot_id, process_step, lot_qty
        FROM lot_history
        WHERE lot_id IN (
            SELECT lot_id FROM lot_master WHERE status = 'ACTIVE'
        )
        AND process_step >= 10
        """,

        # 집계 함수가 포함된 쿼리
        """
        select   area_id, count(*) as cnt, avg(cycle_time) as avg_time
        from     production_log
        where    log_date = '2025-01-25'
        group by area_id
        having   count(*) > 100
        """,
    ]

    print("=" * 80)
    print("sqlglot 기반 SQL 정규화 데모")
    print("=" * 80)

    for i, sql in enumerate(test_sqls, 1):
        print(f"\n{'='*80}")
        print(f"테스트 #{i}")
        print("=" * 80)

        print("\n[원본 SQL]")
        print(sql.strip())

        # 정규화
        normalized = normalize_sql(sql, dialect="oracle")
        print("\n[정규화된 SQL]")
        print(normalized)

        # 템플릿 해시 생성
        template_hash = generate_template_hash(normalized)
        print(f"\n[템플릿 해시]")
        print(f"  SHA-256: {template_hash[:16]}...")

        # 테이블 추출
        tables = extract_tables(sql, dialect="oracle")
        print(f"\n[추출된 테이블]")
        print(f"  {tables}")

        # 컬럼 추출
        columns = extract_columns(sql, dialect="oracle")
        print(f"\n[추출된 컬럼]")
        print(f"  {columns}")

        # WHERE 조건 추출
        conditions = extract_where_conditions(sql, dialect="oracle")
        if conditions:
            print(f"\n[WHERE 조건]")
            for cond in conditions:
                print(f"  - {cond}")

        # JOIN 추출
        joins = extract_joins(sql, dialect="oracle")
        if joins:
            print(f"\n[JOIN 정보]")
            for join in joins:
                print(f"  - {join}")

    # 중복 탐지 예시
    print("\n" + "=" * 80)
    print("중복 탐지 예시")
    print("=" * 80)

    sql1 = "SELECT id, name FROM users WHERE status = 'active'"
    sql2 = "select ID, NAME from USERS where STATUS = 'inactive'"
    sql3 = "SELECT id, name FROM users WHERE status = 123"

    norm1 = normalize_sql(sql1)
    norm2 = normalize_sql(sql2)
    norm3 = normalize_sql(sql3)

    hash1 = generate_template_hash(norm1)
    hash2 = generate_template_hash(norm2)
    hash3 = generate_template_hash(norm3)

    print(f"\nSQL 1: {sql1}")
    print(f"  정규화: {norm1}")
    print(f"  해시: {hash1[:16]}...")

    print(f"\nSQL 2: {sql2}")
    print(f"  정규화: {norm2}")
    print(f"  해시: {hash2[:16]}...")

    print(f"\nSQL 3: {sql3}")
    print(f"  정규화: {norm3}")
    print(f"  해시: {hash3[:16]}...")

    print(f"\n중복 판정 결과:")
    print(f"  SQL 1 == SQL 2: {hash1 == hash2} (리터럴 값만 다름)")
    print(f"  SQL 1 == SQL 3: {hash1 == hash3} (리터럴 타입이 다름)")


if __name__ == "__main__":
    demo()
