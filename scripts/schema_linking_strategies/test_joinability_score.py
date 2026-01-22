#!/usr/bin/env python
"""Joinability Score 기반 스키마 링킹 테스트.

테이블 간 Join 가능성을 점수화하여 관련 테이블을 선정합니다.

점수 요소:
1. 컬럼 중복 (Jaccard): 테이블 간 공통 컬럼명 비율
2. Uniqueness: 컬럼의 고유값 비율 (PK/FK 판별)
3. Subset 관계: 한 컬럼이 다른 컬럼의 부분집합인지
4. 헤더 유사도: 컬럼명 문자열 유사도

사용법:
    python scripts/schema_linking_strategies/test_joinability_score.py --test
    python scripts/schema_linking_strategies/test_joinability_score.py --query "수율과 설비"
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "schema_linking_strategies"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from base import (
    MockSchemaDatabase, SchemaLinkingResult, EvaluationMetrics,
    TEST_CASES, TableInfo, extract_keywords,
    print_result_table, compute_aggregate_metrics,
)


@dataclass
class JoinInfo:
    """테이블 간 Join 정보."""
    table_a: str
    table_b: str
    join_columns: list[str]
    jaccard_score: float
    join_type: str  # "pk_fk", "common_column", "name_similar"


class JoinabilityScorer:
    """테이블 간 Joinability 점수 계산."""

    def __init__(self, schema_db: MockSchemaDatabase):
        self.schema_db = schema_db
        self.join_matrix: dict[str, dict[str, JoinInfo]] = defaultdict(dict)
        self._build_join_matrix()

    def _build_join_matrix(self) -> None:
        """테이블 간 Join 가능성 매트릭스 구축."""
        tables = self.schema_db.get_all_tables()
        
        for i, table_a in enumerate(tables):
            cols_a = set(table_a.column_names)
            pk_cols_a = {c.name for c in table_a.columns if c.is_primary_key}
            fk_cols_a = {c.name for c in table_a.columns if c.is_foreign_key}
            
            for table_b in tables[i+1:]:
                cols_b = set(table_b.column_names)
                common = cols_a & cols_b
                
                if not common:
                    continue
                
                # Jaccard 유사도
                jaccard = len(common) / len(cols_a | cols_b) if (cols_a | cols_b) else 0
                
                # Join 유형 결정
                pk_cols_b = {c.name for c in table_b.columns if c.is_primary_key}
                fk_cols_b = {c.name for c in table_b.columns if c.is_foreign_key}
                
                if common & (pk_cols_a | pk_cols_b | fk_cols_a | fk_cols_b):
                    join_type = "pk_fk"
                else:
                    join_type = "common_column"
                
                join_info = JoinInfo(
                    table_a=table_a.name,
                    table_b=table_b.name,
                    join_columns=list(common),
                    jaccard_score=jaccard,
                    join_type=join_type,
                )
                
                self.join_matrix[table_a.name][table_b.name] = join_info
                self.join_matrix[table_b.name][table_a.name] = join_info

    def get_joinable_tables(self, table_name: str, min_score: float = 0.0) -> list[tuple[str, float]]:
        """특정 테이블과 Join 가능한 테이블 반환."""
        joinable = []
        for other, info in self.join_matrix.get(table_name, {}).items():
            score = info.jaccard_score
            if info.join_type == "pk_fk":
                score += 0.3  # PK/FK 관계 보너스
            if score >= min_score:
                joinable.append((other, score))
        return sorted(joinable, key=lambda x: -x[1])

    def get_join_path(self, table_a: str, table_b: str) -> list[str]:
        """두 테이블 간 Join 경로 찾기 (BFS)."""
        if table_a == table_b:
            return [table_a]
        
        visited = {table_a}
        queue = [(table_a, [table_a])]
        
        while queue:
            current, path = queue.pop(0)
            for neighbor in self.join_matrix.get(current, {}):
                if neighbor == table_b:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []  # 경로 없음


class JoinabilitySchemaLinker:
    """Joinability 기반 스키마 링커."""

    def __init__(self, schema_db: MockSchemaDatabase):
        self.schema_db = schema_db
        self.scorer = JoinabilityScorer(schema_db)

    def link(self, query: str, top_k: int = 5) -> SchemaLinkingResult:
        """Joinability 기반 스키마 링킹."""
        keywords = extract_keywords(query)
        
        # 1단계: 키워드 기반 시드 테이블 선정
        seed_scores = {}
        for table in self.schema_db.get_all_tables():
            score = sum(0.3 if kw.lower() in table.name.lower() else 0 for kw in keywords)
            score += sum(0.2 if kw.lower() in table.description.lower() else 0 for kw in keywords)
            seed_scores[table.name] = score
        
        # 상위 시드 테이블
        sorted_seeds = sorted(seed_scores.items(), key=lambda x: -x[1])
        seed_tables = [t for t, s in sorted_seeds if s > 0][:3]
        
        if not seed_tables:
            seed_tables = [sorted_seeds[0][0]] if sorted_seeds else []
        
        # 2단계: 시드 테이블과 Join 가능한 테이블 확장
        final_scores = dict(seed_scores)
        for seed in seed_tables:
            joinable = self.scorer.get_joinable_tables(seed)
            for table, join_score in joinable:
                boost = join_score * 0.3
                final_scores[table] = final_scores.get(table, 0) + boost
        
        # 정규화
        max_score = max(final_scores.values()) if final_scores else 1.0
        if max_score > 0:
            final_scores = {k: min(v / max_score, 1.0) for k, v in final_scores.items()}
        
        sorted_tables = sorted(final_scores.items(), key=lambda x: -x[1])
        selected = [t[0] for t in sorted_tables[:top_k]]
        
        return SchemaLinkingResult(query=query, selected_tables=selected, scores=final_scores)

    def explain_joins(self, tables: list[str]) -> None:
        """선정된 테이블 간 Join 관계 설명."""
        print("\n[Join 관계]")
        for i, t1 in enumerate(tables):
            for t2 in tables[i+1:]:
                info = self.scorer.join_matrix.get(t1, {}).get(t2)
                if info:
                    print(f"  {t1} <-> {t2}")
                    print(f"    컬럼: {', '.join(info.join_columns)}")
                    print(f"    Jaccard: {info.jaccard_score:.3f}, 유형: {info.join_type}")


def run_tests() -> None:
    print("=" * 70)
    print("Joinability Score 기반 스키마 링킹 테스트")
    print("=" * 70)

    schema_db = MockSchemaDatabase()
    linker = JoinabilitySchemaLinker(schema_db)
    
    # Join Matrix 출력
    print("\n[테이블 Join Matrix]")
    for table_name in list(schema_db.get_table_names())[:5]:
        joinable = linker.scorer.get_joinable_tables(table_name)
        if joinable:
            print(f"\n{table_name}:")
            for other, score in joinable[:3]:
                info = linker.scorer.join_matrix[table_name][other]
                print(f"  -> {other} (score: {score:.3f}, cols: {info.join_columns})")
    
    # 테스트 케이스 실행
    print("\n\n" + "=" * 70)
    print("테스트 케이스 결과")
    print("=" * 70)
    
    results = []
    for tc in TEST_CASES:
        result = linker.link(tc["query"], 5)
        result.ground_truth = tc["ground_truth"]
        results.append(result)
    
    print_result_table(results)
    
    agg = compute_aggregate_metrics(results)
    print("\n📊 집계 지표:")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")


def run_single_query(query: str) -> None:
    print(f"\n질의: {query}")
    print("-" * 60)
    
    schema_db = MockSchemaDatabase()
    linker = JoinabilitySchemaLinker(schema_db)
    result = linker.link(query, 5)
    
    print("\n[선정된 테이블]")
    for i, t in enumerate(result.selected_tables, 1):
        print(f"  {i}. {t} (score: {result.scores[t]:.3f})")
    
    linker.explain_joins(result.selected_tables[:3])


def main():
    parser = argparse.ArgumentParser(description="Joinability Score 테스트")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.query:
        run_single_query(args.query)
    else:
        print("Joinability Score 데모")
        run_single_query("M10 팹의 수율과 설비 정보를 보여줘")


if __name__ == "__main__":
    main()
