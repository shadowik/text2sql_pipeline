#!/usr/bin/env python
"""LLM 기반 스키마 필터링/선택 테스트.

LLM을 활용하여 테이블을 필터링하고 선택하는 전략을 테스트합니다.

전략:
1. Binary Selection (RSL-SQL): 전체/간소 스키마 각각 SQL 생성 → LLM이 더 나은 것 선택
2. Table Purpose Cache (CORE-T): 테이블 목적 메타 + 질의 적합성 판단
3. Question Enrichment (E-SQL): 질의에 엔티티/문맥 추가 후 스키마 선택

사용법:
    python scripts/schema_linking_strategies/test_llm_filtering.py --test
    python scripts/schema_linking_strategies/test_llm_filtering.py --test --use-mock
    python scripts/schema_linking_strategies/test_llm_filtering.py --query "수율 분석"
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
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
# LLM 클라이언트 (Mock + 실제)
# ============================================================================


class MockLLMClient:
    """Mock LLM 클라이언트 (테스트용)."""

    def invoke(self, prompt: str) -> str:
        """키워드 기반 Mock 응답 생성."""
        prompt_lower = prompt.lower()
        
        # Binary Selection 응답
        if "which sql is better" in prompt_lower or "더 나은" in prompt_lower:
            if "full schema" in prompt_lower:
                return "Answer: B (Refined schema SQL is better because it's more focused)"
            return "Answer: A"
        
        # Table Filtering 응답
        if "relevant tables" in prompt_lower or "관련 테이블" in prompt_lower:
            relevant = []
            if "수율" in prompt_lower or "yield" in prompt_lower:
                relevant.append("MES_PRD_YIELD_M10")
            if "설비" in prompt_lower or "equipment" in prompt_lower:
                relevant.append("MES_EQP_MST_M10")
            if "불량" in prompt_lower or "defect" in prompt_lower:
                relevant.append("MES_DEF_HIS_M10")
            if "홀드" in prompt_lower or "hold" in prompt_lower:
                relevant.append("MES_BIZ_LOTHOLD_INF_M10")
            if "공정" in prompt_lower or "process" in prompt_lower:
                relevant.append("MES_PROC_MST_M10")
            if "트래킹" in prompt_lower or "track" in prompt_lower:
                relevant.append("MES_TRK_HIS_M10")
            
            if not relevant:
                relevant = ["MES_PRD_YIELD_M10"]
            
            return f"Relevant tables: {json.dumps(relevant)}"
        
        # Question Enrichment 응답
        if "enrich" in prompt_lower or "확장" in prompt_lower:
            return """Enriched query includes:
            - Entity: fab (M10), yield data
            - Context: production analysis
            - Related: defect rate, equipment status"""
        
        return "No specific response available for this prompt."


def get_llm_client(use_mock: bool = True):
    """LLM 클라이언트 반환."""
    if use_mock:
        return MockLLMClient()
    
    # 실제 LLM 클라이언트 사용
    try:
        from text2sql.core.config import Settings
        from text2sql.adapters.llm.openai_client import OpenAIClient
        return OpenAIClient(Settings())
    except Exception as e:
        print(f"⚠️ 실제 LLM 연결 실패, Mock 사용: {e}")
        return MockLLMClient()


# ============================================================================
# 프롬프트 템플릿
# ============================================================================


PROMPTS = {
    # 테이블 필터링 프롬프트
    "table_filter": """주어진 질의와 관련된 테이블을 선택하세요.

질의: {query}

사용 가능한 테이블:
{table_list}

위 테이블 중에서 질의를 처리하는 데 필요한 테이블만 선택하세요.
JSON 배열 형식으로 테이블명을 반환하세요.

Relevant tables:""",

    # Binary Selection 프롬프트
    "binary_selection": """두 SQL 쿼리 중 더 나은 것을 선택하세요.

질의: {query}

SQL A (Full schema):
{sql_a}

SQL B (Refined schema):
{sql_b}

어떤 SQL이 더 정확하고 효율적인가요?
"Answer: A" 또는 "Answer: B"로 답하세요.

Answer:""",

    # Question Enrichment 프롬프트
    "question_enrichment": """다음 질의를 분석하여 관련 엔티티와 문맥을 추출하세요.

질의: {query}

도메인: 반도체 제조 (MES 시스템)

다음 정보를 추출하세요:
1. 언급된 엔티티 (팹, 설비, 공정 등)
2. 필요한 데이터 유형 (수율, 불량률, 생산량 등)
3. 관련될 수 있는 추가 데이터

Enriched query includes:""",

    # Table Purpose 프롬프트
    "table_purpose": """테이블의 목적과 질의와의 적합성을 평가하세요.

질의: {query}

테이블: {table_name}
설명: {table_desc}
컬럼: {columns}

이 테이블이 질의를 처리하는 데 필요한가요?
1-10 점수와 이유를 제공하세요.

Score:""",
}


# ============================================================================
# Binary Selection (RSL-SQL 스타일)
# ============================================================================


class BinarySelectionLinker:
    """Binary Selection 기반 스키마 링킹.
    
    전체 스키마와 간소화된 스키마로 각각 SQL을 생성하고,
    LLM이 더 나은 결과를 선택합니다.
    """

    def __init__(self, schema_db: MockSchemaDatabase, llm_client):
        self.schema_db = schema_db
        self.llm = llm_client

    def link(self, query: str, top_k: int = 5) -> SchemaLinkingResult:
        """Binary Selection 수행."""
        # 1. 전체 스키마로 SQL 생성 (Mock)
        full_schema_sql = self._generate_sql_with_full_schema(query)
        
        # 2. 간소화된 스키마로 SQL 생성
        refined_tables = self._get_refined_tables(query)
        refined_schema_sql = self._generate_sql_with_refined_schema(query, refined_tables)
        
        # 3. LLM에게 선택 요청
        selected = self._ask_llm_to_select(query, full_schema_sql, refined_schema_sql)
        
        # 4. 선택된 SQL에서 테이블 추출
        if selected == "B":
            final_tables = refined_tables
        else:
            final_tables = self._extract_tables(full_schema_sql)
        
        # 점수 계산 (선택된 테이블에 높은 점수)
        scores = {}
        for i, table_name in enumerate(self.schema_db.get_table_names()):
            if table_name in final_tables:
                scores[table_name] = 1.0 - (final_tables.index(table_name) * 0.1)
            else:
                scores[table_name] = 0.1
        
        return SchemaLinkingResult(
            query=query,
            selected_tables=final_tables[:top_k],
            scores=scores,
        )

    def _generate_sql_with_full_schema(self, query: str) -> str:
        """전체 스키마로 SQL 생성 (Mock)."""
        # 간단한 키워드 기반 SQL 생성
        keywords = extract_keywords(query)
        tables = []
        
        for table in self.schema_db.get_all_tables():
            if any(kw.lower() in table.name.lower() or 
                   kw.lower() in table.description.lower() 
                   for kw in keywords):
                tables.append(table.name)
        
        if not tables:
            tables = [self.schema_db.get_all_tables()[0].name]
        
        return f"SELECT * FROM {tables[0]} -- full schema"

    def _get_refined_tables(self, query: str) -> list[str]:
        """키워드 기반 테이블 정제."""
        keywords = extract_keywords(query)
        scored_tables = []
        
        for table in self.schema_db.get_all_tables():
            score = 0
            for kw in keywords:
                if kw.lower() in table.name.lower():
                    score += 2
                if kw.lower() in table.description.lower():
                    score += 1
            if score > 0:
                scored_tables.append((table.name, score))
        
        scored_tables.sort(key=lambda x: -x[1])
        return [t[0] for t in scored_tables[:5]]

    def _generate_sql_with_refined_schema(self, query: str, tables: list[str]) -> str:
        """정제된 스키마로 SQL 생성 (Mock)."""
        if not tables:
            return "SELECT * FROM UNKNOWN -- refined"
        return f"SELECT * FROM {tables[0]} -- refined schema"

    def _ask_llm_to_select(self, query: str, sql_a: str, sql_b: str) -> str:
        """LLM에게 더 나은 SQL 선택 요청."""
        prompt = PROMPTS["binary_selection"].format(
            query=query, sql_a=sql_a, sql_b=sql_b
        )
        response = self.llm.invoke(prompt)
        
        if "B" in response.upper():
            return "B"
        return "A"

    def _extract_tables(self, sql: str) -> list[str]:
        """SQL에서 테이블명 추출."""
        matches = re.findall(r"FROM\s+([A-Za-z0-9_]+)", sql, re.IGNORECASE)
        matches += re.findall(r"JOIN\s+([A-Za-z0-9_]+)", sql, re.IGNORECASE)
        return list(dict.fromkeys(matches))  # 순서 유지하며 중복 제거


# ============================================================================
# Table Purpose Cache (CORE-T 스타일)
# ============================================================================


class TablePurposeCacheLinker:
    """Table Purpose Cache 기반 스키마 링킹.
    
    테이블의 목적 메타데이터와 질의 적합성을 판단합니다.
    """

    def __init__(self, schema_db: MockSchemaDatabase, llm_client):
        self.schema_db = schema_db
        self.llm = llm_client
        self.purpose_cache: dict[str, dict] = {}
        self._build_purpose_cache()

    def _build_purpose_cache(self) -> None:
        """테이블 목적 캐시 구축."""
        for table in self.schema_db.get_all_tables():
            self.purpose_cache[table.name] = {
                "purpose": table.purpose or table.description,
                "keywords": self._extract_purpose_keywords(table),
                "data_type": self._infer_data_type(table),
            }

    def _extract_purpose_keywords(self, table: TableInfo) -> list[str]:
        """테이블 목적에서 키워드 추출."""
        text = f"{table.name} {table.description} {table.purpose}"
        return extract_keywords(text)

    def _infer_data_type(self, table: TableInfo) -> str:
        """테이블의 데이터 유형 추론."""
        name = table.name.upper()
        if "MST" in name:
            return "master"
        if "HIS" in name:
            return "history"
        if "INF" in name:
            return "information"
        if "TRK" in name:
            return "tracking"
        return "data"

    def link(self, query: str, top_k: int = 5) -> SchemaLinkingResult:
        """Table Purpose 기반 링킹 수행."""
        query_keywords = set(extract_keywords(query))
        
        scores = {}
        for table_name, cache in self.purpose_cache.items():
            # 키워드 매칭 점수
            purpose_keywords = set(cache["keywords"])
            keyword_overlap = len(query_keywords & purpose_keywords)
            keyword_score = keyword_overlap / max(len(query_keywords), 1)
            
            # 데이터 유형 보정
            data_type_boost = 0.0
            if cache["data_type"] == "history":
                data_type_boost = 0.1  # 이력 테이블 선호
            
            scores[table_name] = keyword_score + data_type_boost
        
        # 상위 k개 선택
        sorted_tables = sorted(scores.items(), key=lambda x: -x[1])
        selected = [t[0] for t in sorted_tables[:top_k]]
        
        return SchemaLinkingResult(
            query=query,
            selected_tables=selected,
            scores=scores,
        )


# ============================================================================
# Question Enrichment (E-SQL 스타일)
# ============================================================================


class QuestionEnrichmentLinker:
    """Question Enrichment 기반 스키마 링킹.
    
    질의에 엔티티와 문맥을 추가하여 스키마 선택 정확도를 높입니다.
    """

    def __init__(self, schema_db: MockSchemaDatabase, llm_client):
        self.schema_db = schema_db
        self.llm = llm_client

    def link(self, query: str, top_k: int = 5) -> SchemaLinkingResult:
        """Question Enrichment 수행."""
        # 1. 질의 강화
        enriched_info = self._enrich_question(query)
        
        # 2. 강화된 정보로 테이블 매칭
        scores = self._match_with_enriched_query(query, enriched_info)
        
        # 상위 k개 선택
        sorted_tables = sorted(scores.items(), key=lambda x: -x[1])
        selected = [t[0] for t in sorted_tables[:top_k]]
        
        return SchemaLinkingResult(
            query=query,
            selected_tables=selected,
            scores=scores,
        )

    def _enrich_question(self, query: str) -> dict:
        """LLM을 사용하여 질의 강화."""
        prompt = PROMPTS["question_enrichment"].format(query=query)
        response = self.llm.invoke(prompt)
        
        # 응답 파싱 (간단한 키워드 추출)
        return {
            "entities": self._extract_entities(response),
            "data_types": self._extract_data_types(response),
            "original_keywords": extract_keywords(query),
        }

    def _extract_entities(self, text: str) -> list[str]:
        """텍스트에서 엔티티 추출."""
        entities = []
        # 팹 ID 추출
        fab_match = re.findall(r"(M10|M11|M14|M15|M16)", text, re.IGNORECASE)
        entities.extend(fab_match)
        
        # 키워드 추출
        keywords = extract_keywords(text)
        entities.extend(keywords)
        
        return list(set(entities))

    def _extract_data_types(self, text: str) -> list[str]:
        """필요한 데이터 유형 추출."""
        data_types = []
        type_keywords = {
            "yield": "수율",
            "defect": "불량",
            "equipment": "설비",
            "production": "생산",
            "tracking": "트래킹",
        }
        
        for eng, kor in type_keywords.items():
            if eng in text.lower() or kor in text:
                data_types.append(eng)
        
        return data_types

    def _match_with_enriched_query(self, query: str, enriched: dict) -> dict[str, float]:
        """강화된 질의로 테이블 매칭."""
        all_keywords = set(enriched["original_keywords"])
        all_keywords.update(enriched["entities"])
        
        # 데이터 유형 기반 키워드 추가
        type_to_keywords = {
            "yield": ["yield", "수율", "PRD_YIELD"],
            "defect": ["defect", "불량", "DEF"],
            "equipment": ["equipment", "설비", "EQP"],
            "production": ["생산", "PRD"],
            "tracking": ["tracking", "트래킹", "TRK"],
        }
        
        for data_type in enriched.get("data_types", []):
            if data_type in type_to_keywords:
                all_keywords.update(type_to_keywords[data_type])
        
        scores = {}
        for table in self.schema_db.get_all_tables():
            score = 0.0
            table_text = f"{table.name} {table.description} {table.purpose}"
            
            for kw in all_keywords:
                if kw.lower() in table_text.lower():
                    score += 0.2
            
            scores[table.name] = min(score, 1.0)
        
        return scores


# ============================================================================
# 통합 LLM 스키마 링커
# ============================================================================


class LLMSchemaLinker:
    """LLM 기반 통합 스키마 링커."""

    def __init__(
        self,
        schema_db: MockSchemaDatabase,
        use_mock_llm: bool = True,
        strategy: str = "table_purpose",
    ):
        self.schema_db = schema_db
        self.llm = get_llm_client(use_mock_llm)
        self.strategy = strategy
        
        # 전략별 링커 초기화
        self.linkers = {
            "binary_selection": BinarySelectionLinker(schema_db, self.llm),
            "table_purpose": TablePurposeCacheLinker(schema_db, self.llm),
            "question_enrichment": QuestionEnrichmentLinker(schema_db, self.llm),
        }

    def link(self, query: str, top_k: int = 5) -> SchemaLinkingResult:
        """선택된 전략으로 스키마 링킹 수행."""
        linker = self.linkers.get(self.strategy, self.linkers["table_purpose"])
        return linker.link(query, top_k)


# ============================================================================
# 테스트 실행
# ============================================================================


def run_tests(use_mock: bool = True) -> None:
    """테스트 케이스 실행."""
    print("=" * 80)
    print("LLM 기반 스키마 필터링 테스트")
    print(f"LLM: {'Mock' if use_mock else 'Real'}")
    print("=" * 80)
    
    schema_db = MockSchemaDatabase()
    
    strategies = ["binary_selection", "table_purpose", "question_enrichment"]
    
    for strategy in strategies:
        print(f"\n\n{'='*40}")
        print(f"전략: {strategy}")
        print(f"{'='*40}")
        
        linker = LLMSchemaLinker(
            schema_db=schema_db,
            use_mock_llm=use_mock,
            strategy=strategy,
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


def run_single_query(query: str, strategy: str = "table_purpose", use_mock: bool = True) -> None:
    """단일 질의 테스트."""
    print(f"\n질의: {query}")
    print(f"전략: {strategy}")
    print(f"LLM: {'Mock' if use_mock else 'Real'}")
    print("-" * 60)
    
    schema_db = MockSchemaDatabase()
    linker = LLMSchemaLinker(
        schema_db=schema_db,
        use_mock_llm=use_mock,
        strategy=strategy,
    )
    
    result = linker.link(query, top_k=5)
    
    print("\n선정된 테이블:")
    for i, table in enumerate(result.selected_tables, 1):
        score = result.scores.get(table, 0.0)
        print(f"  {i}. {table} (score: {score:.4f})")


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="LLM 기반 스키마 필터링 테스트",
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
        "--strategy",
        type=str,
        choices=["binary_selection", "table_purpose", "question_enrichment"],
        default="table_purpose",
        help="사용할 전략 (기본값: table_purpose)",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        default=True,
        help="Mock LLM 사용 (기본값: True)",
    )
    parser.add_argument(
        "--use-real-llm",
        action="store_true",
        help="실제 LLM 사용",
    )
    
    args = parser.parse_args()
    use_mock = not args.use_real_llm
    
    if args.test:
        run_tests(use_mock)
    elif args.query:
        run_single_query(args.query, args.strategy, use_mock)
    else:
        print("LLM 기반 스키마 필터링 데모")
        print("-" * 40)
        run_single_query("M10 팹의 수율 데이터를 보여줘", "table_purpose", use_mock)


if __name__ == "__main__":
    main()
