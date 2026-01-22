#!/usr/bin/env python
"""RAG 메타데이터 증강 기반 스키마 링킹 테스트.

증강 요소: 용어 사전, 샘플 값, SQL 로그 Few-shot

사용법:
    python scripts/schema_linking_strategies/test_rag_augmentation.py --test
    python scripts/schema_linking_strategies/test_rag_augmentation.py --query "수율 분석"
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "schema_linking_strategies"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from base import (
    MockSchemaDatabase, SchemaLinkingResult, EvaluationMetrics,
    TEST_CASES, TableInfo, GlossaryTerm, extract_keywords,
    print_result_table, compute_aggregate_metrics,
)


class GlossaryLoader:
    """용어 사전 로더."""

    def __init__(self, path: Optional[Path] = None):
        self.terms: dict[str, GlossaryTerm] = {}
        self.korean_to_english: dict[str, str] = {}
        if path and path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                term = GlossaryTerm(
                    term=row.get("term", ""),
                    korean_name=row.get("korean_name", ""),
                    description=row.get("description", ""),
                    category=row.get("category", ""),
                )
                self.terms[term.term.lower()] = term
                if term.korean_name:
                    self.korean_to_english[term.korean_name] = term.term

    def expand_query(self, query: str) -> list[str]:
        """질의 키워드 확장."""
        keywords = extract_keywords(query)
        expanded = set(keywords)
        for kw in keywords:
            if kw in self.korean_to_english:
                expanded.add(self.korean_to_english[kw])
            for t in self.terms.values():
                if kw.lower() in t.term or kw.lower() in t.description.lower():
                    expanded.add(t.term)
        return list(expanded)


class SQLLogAnalyzer:
    """SQL 로그 분석."""

    def __init__(self, path: Optional[Path] = None):
        self.logs: list[dict] = []
        self.table_usage: Counter = Counter()
        if path and path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            self.logs = json.load(f)
        for log in self.logs:
            for table in self._extract_tables(log.get("sql_text", "")):
                self.table_usage[table] += log.get("exec_count", 1)

    def _extract_tables(self, sql: str) -> list[str]:
        tables = re.findall(r"FROM\s+([A-Za-z0-9_]+)", sql, re.I)
        tables += re.findall(r"JOIN\s+([A-Za-z0-9_]+)", sql, re.I)
        return list(set(tables))

    def find_similar(self, keywords: list[str], top_k: int = 3) -> list[list[str]]:
        """유사 쿼리의 테이블 반환."""
        results = []
        for log in self.logs:
            sql = log.get("sql_text", "").lower()
            if any(kw.lower() in sql for kw in keywords):
                results.append(self._extract_tables(log.get("sql_text", "")))
        return results[:top_k]


class RAGSchemaLinker:
    """RAG 기반 스키마 링커."""

    def __init__(self, schema_db: MockSchemaDatabase, use_glossary=True, use_fewshot=True):
        self.schema_db = schema_db
        self.glossary = GlossaryLoader(PROJECT_ROOT / "data/samples/glossary.csv") if use_glossary else None
        self.sql_analyzer = SQLLogAnalyzer(PROJECT_ROOT / "data/samples/sql_logs.json") if use_fewshot else None

    def link(self, query: str, top_k: int = 5) -> SchemaLinkingResult:
        keywords = set(extract_keywords(query))
        if self.glossary:
            keywords.update(self.glossary.expand_query(query))

        # 키워드 매칭 점수
        scores = {}
        for table in self.schema_db.get_all_tables():
            score = sum(0.3 if kw.lower() in table.name.lower() else 0 for kw in keywords)
            score += sum(0.2 if kw.lower() in table.description.lower() else 0 for kw in keywords)
            for col in table.columns:
                score += sum(0.1 if kw.lower() in col.name.lower() else 0 for kw in keywords)
            scores[table.name] = min(score, 1.0)

        # Few-shot 보정
        if self.sql_analyzer:
            for tables in self.sql_analyzer.find_similar(list(keywords)):
                for t in tables:
                    if t in scores:
                        scores[t] = min(scores[t] + 0.15, 1.0)

        sorted_tables = sorted(scores.items(), key=lambda x: -x[1])
        return SchemaLinkingResult(query=query, selected_tables=[t[0] for t in sorted_tables[:top_k]], scores=scores)


def run_tests() -> None:
    print("=" * 70)
    print("RAG 메타데이터 증강 테스트")
    print("=" * 70)

    schema_db = MockSchemaDatabase()
    configs = [("No RAG", False, False), ("Glossary", True, False), ("Few-shot", False, True), ("Full RAG", True, True)]

    for name, use_g, use_f in configs:
        print(f"\n{'='*40}\n설정: {name}\n{'='*40}")
        linker = RAGSchemaLinker(schema_db, use_glossary=use_g, use_fewshot=use_f)
        results = [linker.link(tc["query"], 5) for tc in TEST_CASES]
        for r, tc in zip(results, TEST_CASES):
            r.ground_truth = tc["ground_truth"]
        print_result_table(results)
        agg = compute_aggregate_metrics(results)
        print("📊 지표:", {k: f"{v:.3f}" for k, v in agg.items()})


def main():
    parser = argparse.ArgumentParser(description="RAG 스키마 링킹 테스트")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.query:
        linker = RAGSchemaLinker(MockSchemaDatabase())
        result = linker.link(args.query, 5)
        print(f"\n질의: {args.query}\n선정 테이블:")
        for i, t in enumerate(result.selected_tables, 1):
            print(f"  {i}. {t} ({result.scores[t]:.3f})")
    else:
        run_tests()


if __name__ == "__main__":
    main()
