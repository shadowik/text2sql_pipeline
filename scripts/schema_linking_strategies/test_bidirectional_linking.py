#!/usr/bin/env python
"""양방향 스키마 링킹 테스트.

Forward Linking과 Backward Linking을 결합하여 recall을 높입니다.

전략:
1. Forward Linking: 질의 → 스키마 직접 매칭으로 후보 테이블 선정
2. Backward Linking: 초기 SQL 생성 → 사용된 테이블/컬럼 추출
3. SQL-to-Schema: 전체 스키마로 SQL 생성 후 필요 스키마만 추출해 재생성

사용법:
    python scripts/schema_linking_strategies/test_bidirectional_linking.py --test
    python scripts/schema_linking_strategies/test_bidirectional_linking.py --query "수율 분석"
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "schema_linking_strategies"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from base import (
    MockSchemaDatabase,
    SchemaLinkingResult,
    EvaluationMetrics,
    TEST_CASES,
    TableInfo,
    extract_keywords,
    print_result_table,
    compute_aggregate_metrics,
)


# ============================================================================
# Forward Linking
# ============================================================================


class ForwardLinker:
    """Forward Linking: 질의에서 스키마로 직접 매칭."""

    def __init__(self, schema_db: MockSchemaDatabase):
        self.schema_db = schema_db
        self._load_glossary()

    def _load_glossary(self) -> None:
        """용어 사전 로드."""
        glossary_path = PROJECT_ROOT / "data" / "samples" / "glossary.csv"
        self.schema_db.load_glossary(glossary_path)

    def link(self, query: str) -> dict[str, float]:
        """질의와 스키마 직접 매칭.
        
        Args:
            query: 자연어 질의
            
        Returns:
            테이블별 매칭 점수
        """
        keywords = extract_keywords(query)
        
        # 용어 사전으로 키워드 확장
        expanded_keywords = set(keywords)
        for kw in keywords:
            synonyms = self.schema_db.find_synonyms(kw)
            expanded_keywords.update(synonyms)
        
        scores = {}
        for table in self.schema_db.get_all_tables():
            score = self._compute_match_score(table, expanded_keywords)
            scores[table.name] = score
        
        return scores

    def _compute_match_score(self, table: TableInfo, keywords: set[str]) -> float:
        """테이블과 키워드 간 매칭 점수 계산."""
        score = 0.0
        
        # 테이블명 매칭
        table_name_lower = table.name.lower()
        for kw in keywords:
            if kw.lower() in table_name_lower:
                score += 0.3
        
        # 설명 매칭
        desc_lower = (table.description + " " + table.purpose).lower()
        for kw in keywords:
            if kw.lower() in desc_lower:
                score += 0.2
        
        # 컬럼명 매칭
        for col in table.columns:
            col_name_lower = col.name.lower()
            for kw in keywords:
                if kw.lower() in col_name_lower:
                    score += 0.15
            
            # 컬럼 설명 매칭
            col_desc_lower = col.description.lower()
            for kw in keywords:
                if kw.lower() in col_desc_lower:
                    score += 0.1
        
        return min(score, 1.0)


# ============================================================================
# Backward Linking (SQL-to-Schema)
# ============================================================================


class BackwardLinker:
    """Backward Linking: SQL에서 테이블/컬럼 추출."""

    def __init__(self, schema_db: MockSchemaDatabase):
        self.schema_db = schema_db

    def extract_tables_from_sql(self, sql: str) -> list[str]:
        """SQL에서 사용된 테이블 추출.
        
        Args:
            sql: SQL 문
            
        Returns:
            추출된 테이블명 리스트
        """
        # FROM, JOIN 절에서 테이블명 추출
        patterns = [
            r"FROM\s+([A-Za-z0-9_]+)",
            r"JOIN\s+([A-Za-z0-9_]+)",
            r"INTO\s+([A-Za-z0-9_]+)",
            r"UPDATE\s+([A-Za-z0-9_]+)",
        ]
        
        tables = set()
        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            tables.update(matches)
        
        # 유효한 테이블명만 필터링
        valid_tables = set(self.schema_db.get_table_names())
        return [t for t in tables if t.upper() in (vt.upper() for vt in valid_tables)]

    def extract_columns_from_sql(self, sql: str) -> list[str]:
        """SQL에서 사용된 컬럼 추출."""
        # SELECT, WHERE, GROUP BY, ORDER BY 등에서 컬럼 추출
        # 단순화된 구현 - 실제로는 SQL 파서 사용 권장
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql)
        
        # 키워드 제외
        keywords = {
            "SELECT", "FROM", "WHERE", "AND", "OR", "JOIN", "LEFT", "RIGHT",
            "INNER", "OUTER", "ON", "AS", "IN", "NOT", "NULL", "IS", "BETWEEN",
            "GROUP", "BY", "ORDER", "ASC", "DESC", "HAVING", "LIMIT", "COUNT",
            "SUM", "AVG", "MAX", "MIN", "DISTINCT", "CASE", "WHEN", "THEN",
            "ELSE", "END", "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE",
            "CREATE", "TABLE", "DROP", "ALTER", "INDEX", "TO_DATE", "SYSDATE",
            "TRUNC", "ROUND", "NVL", "DECODE", "LIKE", "OVER", "PARTITION",
        }
        
        return [w for w in words if w.upper() not in keywords]


class MockSQLGenerator:
    """Mock SQL 생성기.
    
    실제 환경에서는 LLM을 사용하여 SQL을 생성합니다.
    """

    def __init__(self, schema_db: MockSchemaDatabase):
        self.schema_db = schema_db

    def generate(self, query: str, available_tables: list[str] = None) -> str:
        """질의로부터 SQL 생성 (Mock).
        
        Args:
            query: 자연어 질의
            available_tables: 사용 가능한 테이블 리스트 (None이면 전체)
            
        Returns:
            생성된 SQL (Mock)
        """
        # 간단한 키워드 기반 SQL 생성
        keywords = extract_keywords(query)
        
        # 키워드와 매칭되는 테이블 찾기
        matched_tables = []
        for table in self.schema_db.get_all_tables():
            if available_tables and table.name not in available_tables:
                continue
            
            if any(kw.lower() in table.name.lower() or 
                   kw.lower() in table.description.lower() 
                   for kw in keywords):
                matched_tables.append(table)
        
        if not matched_tables:
            # 기본 테이블 반환
            all_tables = self.schema_db.get_all_tables()
            if available_tables:
                matched_tables = [t for t in all_tables if t.name in available_tables][:1]
            else:
                matched_tables = all_tables[:1]
        
        # Mock SQL 생성
        if matched_tables:
            main_table = matched_tables[0]
            columns = ", ".join(main_table.column_names[:5])
            
            sql = f"SELECT {columns} FROM {main_table.name}"
            
            # JOIN 추가 (다중 테이블인 경우)
            if len(matched_tables) > 1:
                for join_table in matched_tables[1:3]:
                    # 공통 컬럼 찾기
                    common_cols = set(main_table.column_names) & set(join_table.column_names)
                    if common_cols:
                        join_col = list(common_cols)[0]
                        sql += f" JOIN {join_table.name} ON {main_table.name}.{join_col} = {join_table.name}.{join_col}"
            
            return sql
        
        return "SELECT * FROM UNKNOWN_TABLE"


# ============================================================================
# 양방향 스키마 링커
# ============================================================================


class BidirectionalSchemaLinker:
    """양방향 스키마 링킹.
    
    Forward + Backward 링킹을 결합하여 recall을 높입니다.
    """

    def __init__(
        self,
        schema_db: MockSchemaDatabase,
        forward_weight: float = 0.6,
        use_sql_refinement: bool = True,
    ):
        self.schema_db = schema_db
        self.forward_weight = forward_weight
        self.backward_weight = 1.0 - forward_weight
        self.use_sql_refinement = use_sql_refinement
        
        self.forward_linker = ForwardLinker(schema_db)
        self.backward_linker = BackwardLinker(schema_db)
        self.sql_generator = MockSQLGenerator(schema_db)

    def link(self, query: str, top_k: int = 5) -> SchemaLinkingResult:
        """양방향 스키마 링킹 수행.
        
        Args:
            query: 자연어 질의
            top_k: 반환할 상위 테이블 수
            
        Returns:
            SchemaLinkingResult
        """
        # 1. Forward Linking
        forward_scores = self.forward_linker.link(query)
        
        # 2. Backward Linking (SQL 생성 → 테이블 추출)
        # 먼저 전체 스키마로 SQL 생성
        initial_sql = self.sql_generator.generate(query)
        backward_tables = self.backward_linker.extract_tables_from_sql(initial_sql)
        
        # Backward 점수 계산
        backward_scores = {}
        for table_name in self.schema_db.get_table_names():
            if table_name in backward_tables:
                backward_scores[table_name] = 1.0
            else:
                backward_scores[table_name] = 0.0
        
        # 3. 점수 융합
        combined_scores = {}
        for table_name in self.schema_db.get_table_names():
            fwd_score = forward_scores.get(table_name, 0.0)
            bwd_score = backward_scores.get(table_name, 0.0)
            combined_scores[table_name] = (
                self.forward_weight * fwd_score + self.backward_weight * bwd_score
            )
        
        # 4. SQL-to-Schema 정제 (옵션)
        if self.use_sql_refinement:
            # 상위 테이블만 사용해서 SQL 재생성
            top_tables = sorted(combined_scores.items(), key=lambda x: -x[1])[:top_k + 2]
            top_table_names = [t[0] for t in top_tables]
            
            refined_sql = self.sql_generator.generate(query, top_table_names)
            refined_tables = self.backward_linker.extract_tables_from_sql(refined_sql)
            
            # 정제된 결과로 점수 조정
            for table_name in refined_tables:
                if table_name in combined_scores:
                    combined_scores[table_name] += 0.2
        
        # 상위 k개 선택
        sorted_tables = sorted(combined_scores.items(), key=lambda x: -x[1])
        selected = [t[0] for t in sorted_tables[:top_k]]
        
        return SchemaLinkingResult(
            query=query,
            selected_tables=selected,
            scores=combined_scores,
        )


# ============================================================================
# 테스트 실행
# ============================================================================


def run_tests() -> None:
    """테스트 케이스 실행."""
    print("=" * 80)
    print("양방향 스키마 링킹 테스트")
    print("=" * 80)
    
    schema_db = MockSchemaDatabase()
    
    # 전략별 비교
    strategies = [
        ("Forward Only", 1.0, False),
        ("Backward Only", 0.0, False),
        ("Bidirectional (0.6:0.4)", 0.6, False),
        ("Bidirectional + Refinement", 0.6, True),
    ]
    
    for strategy_name, forward_weight, use_refinement in strategies:
        print(f"\n\n{'='*40}")
        print(f"전략: {strategy_name}")
        print(f"{'='*40}")
        
        linker = BidirectionalSchemaLinker(
            schema_db=schema_db,
            forward_weight=forward_weight,
            use_sql_refinement=use_refinement,
        )
        
        results = []
        for test_case in TEST_CASES:
            result = linker.link(test_case["query"], top_k=5)
            result.ground_truth = test_case["ground_truth"]
            results.append(result)
        
        print_result_table(results)
        
        # 집계 지표
        aggregate = compute_aggregate_metrics(results)
        print("\n📊 집계 지표:")
        for metric, value in aggregate.items():
            print(f"  {metric}: {value:.4f}")


def run_single_query(query: str, forward_weight: float = 0.6, use_refinement: bool = True) -> None:
    """단일 질의 테스트."""
    print(f"\n질의: {query}")
    print(f"가중치: Forward={forward_weight}, Backward={1-forward_weight}")
    print(f"SQL 정제: {'활성화' if use_refinement else '비활성화'}")
    print("-" * 60)
    
    schema_db = MockSchemaDatabase()
    linker = BidirectionalSchemaLinker(
        schema_db=schema_db,
        forward_weight=forward_weight,
        use_sql_refinement=use_refinement,
    )
    
    # Forward/Backward 개별 결과도 출력
    forward_scores = linker.forward_linker.link(query)
    initial_sql = linker.sql_generator.generate(query)
    backward_tables = linker.backward_linker.extract_tables_from_sql(initial_sql)
    
    print("\n[Forward Linking 결과]")
    top_forward = sorted(forward_scores.items(), key=lambda x: -x[1])[:3]
    for table, score in top_forward:
        print(f"  - {table}: {score:.4f}")
    
    print(f"\n[Backward Linking 결과]")
    print(f"  생성된 SQL: {initial_sql[:100]}...")
    print(f"  추출된 테이블: {backward_tables}")
    
    # 최종 결과
    result = linker.link(query, top_k=5)
    
    print("\n[최종 선정 테이블]")
    for i, table in enumerate(result.selected_tables, 1):
        score = result.scores.get(table, 0.0)
        print(f"  {i}. {table} (score: {score:.4f})")


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="양방향 스키마 링킹 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="전체 테스트 케이스 실행",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="테스트할 자연어 질의",
    )
    parser.add_argument(
        "--forward-weight",
        type=float,
        default=0.6,
        help="Forward 링킹 가중치 (기본값: 0.6)",
    )
    parser.add_argument(
        "--no-refinement",
        action="store_true",
        help="SQL 정제 비활성화",
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.query:
        run_single_query(args.query, args.forward_weight, not args.no_refinement)
    else:
        print("양방향 스키마 링킹 데모")
        print("-" * 40)
        run_single_query("M10 팹의 수율 데이터를 보여줘")


if __name__ == "__main__":
    main()
