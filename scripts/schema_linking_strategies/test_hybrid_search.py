#!/usr/bin/env python
"""하이브리드 검색 기반 스키마 링킹 테스트.

벡터 검색(의미 유사도) + BM25 키워드 검색을 융합하여 top-k 테이블을 선정합니다.

전략:
1. Vector Search: 질의 임베딩과 테이블/컬럼 설명 임베딩 간 코사인 유사도
2. BM25 Search: 테이블명, 컬럼명 키워드 기반 검색
3. Hybrid Fusion: 가중치 융합 (Vector α + BM25 (1-α))
4. Re-ranking: LLM 기반 재순위화 (옵션)

사용법:
    python scripts/schema_linking_strategies/test_hybrid_search.py --test
    python scripts/schema_linking_strategies/test_hybrid_search.py --query "수율 데이터 조회"
"""

import argparse
import math
import re
import sys
from collections import Counter
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
# 간단한 임베딩 시뮬레이션 (실제 환경에서는 실제 임베딩 사용)
# ============================================================================


class SimpleEmbedding:
    """간단한 TF-IDF 기반 임베딩 시뮬레이션.
    
    실제 환경에서는 OpenAI/로컬 임베딩 모델로 대체해야 합니다.
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.docs: list[list[str]] = []

    def fit(self, documents: list[str]) -> None:
        """문서로부터 어휘와 IDF 계산."""
        self.docs = [self._tokenize(doc) for doc in documents]
        
        # 어휘 구축
        all_words = set()
        for tokens in self.docs:
            all_words.update(tokens)
        self.vocab = {word: i for i, word in enumerate(sorted(all_words))}
        
        # IDF 계산
        n_docs = len(self.docs)
        for word in self.vocab:
            doc_count = sum(1 for tokens in self.docs if word in tokens)
            self.idf[word] = math.log((n_docs + 1) / (doc_count + 1)) + 1

    def embed(self, text: str) -> list[float]:
        """텍스트를 TF-IDF 벡터로 변환."""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        
        vector = [0.0] * len(self.vocab)
        for word, count in tf.items():
            if word in self.vocab:
                idx = self.vocab[word]
                vector[idx] = count * self.idf.get(word, 1.0)
        
        # L2 정규화
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector

    def _tokenize(self, text: str) -> list[str]:
        """텍스트 토큰화."""
        words = re.findall(r"[가-힣]+|[A-Za-z0-9_]+", text.lower())
        return words

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """코사인 유사도 계산."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


# ============================================================================
# BM25 검색
# ============================================================================


class BM25:
    """BM25 키워드 검색."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_len: float = 0.0
        self.idf: dict[str, float] = {}

    def fit(self, documents: list[str]) -> None:
        """문서 인덱싱."""
        self.docs = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.docs]
        self.avg_doc_len = sum(self.doc_lengths) / len(self.docs) if self.docs else 0
        
        # IDF 계산
        n_docs = len(self.docs)
        all_words = set()
        for tokens in self.docs:
            all_words.update(tokens)
        
        for word in all_words:
            doc_count = sum(1 for tokens in self.docs if word in tokens)
            self.idf[word] = math.log((n_docs - doc_count + 0.5) / (doc_count + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        """쿼리와 문서 간 BM25 점수 계산."""
        query_tokens = self._tokenize(query)
        doc_tokens = self.docs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0.0
        tf = Counter(doc_tokens)
        
        for term in query_tokens:
            if term not in self.idf:
                continue
            
            term_freq = tf.get(term, 0)
            idf = self.idf[term]
            
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            
            score += idf * numerator / denominator
        
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """쿼리로 문서 검색."""
        scores = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """텍스트 토큰화."""
        words = re.findall(r"[가-힣]+|[A-Za-z0-9_]+", text.lower())
        return words


# ============================================================================
# 하이브리드 검색 스키마 링커
# ============================================================================


class HybridSchemaLinker:
    """하이브리드 검색 기반 스키마 링커.
    
    Vector + BM25 융합으로 테이블을 선정합니다.
    """

    def __init__(
        self,
        schema_db: MockSchemaDatabase,
        vector_weight: float = 0.7,
        use_reranking: bool = False,
    ):
        self.schema_db = schema_db
        self.vector_weight = vector_weight
        self.bm25_weight = 1.0 - vector_weight
        self.use_reranking = use_reranking
        
        self.embedding = SimpleEmbedding()
        self.bm25 = BM25()
        self.table_names: list[str] = []
        self.table_docs: list[str] = []
        
        self._build_index()

    def _build_index(self) -> None:
        """테이블 인덱스 구축."""
        self.table_names = []
        self.table_docs = []
        
        for table in self.schema_db.get_all_tables():
            self.table_names.append(table.name)
            
            # 테이블 문서 생성: 이름 + 설명 + 컬럼 정보
            doc_parts = [
                table.name,
                table.description,
                table.purpose,
            ]
            for col in table.columns:
                doc_parts.extend([col.name, col.description])
                doc_parts.extend(col.sample_values)
            
            self.table_docs.append(" ".join(doc_parts))
        
        # 임베딩 및 BM25 인덱스 구축
        self.embedding.fit(self.table_docs)
        self.bm25.fit(self.table_docs)

    def link(self, query: str, top_k: int = 5) -> SchemaLinkingResult:
        """질의에 대한 스키마 링킹 수행.
        
        Args:
            query: 자연어 질의
            top_k: 반환할 상위 테이블 수
            
        Returns:
            SchemaLinkingResult
        """
        # 1. Vector Search
        query_vec = self.embedding.embed(query)
        vector_scores = {}
        
        for i, table_name in enumerate(self.table_names):
            table_vec = self.embedding.embed(self.table_docs[i])
            score = SimpleEmbedding.cosine_similarity(query_vec, table_vec)
            vector_scores[table_name] = score

        # 2. BM25 Search
        bm25_results = self.bm25.search(query, top_k=len(self.table_names))
        bm25_scores = {self.table_names[idx]: score for idx, score in bm25_results}
        
        # 점수 정규화
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        if max_bm25 > 0:
            bm25_scores = {k: v / max_bm25 for k, v in bm25_scores.items()}

        # 3. Hybrid Fusion
        hybrid_scores = {}
        for table_name in self.table_names:
            vec_score = vector_scores.get(table_name, 0.0)
            bm25_score = bm25_scores.get(table_name, 0.0)
            hybrid_scores[table_name] = (
                self.vector_weight * vec_score + self.bm25_weight * bm25_score
            )

        # 4. Re-ranking (옵션)
        if self.use_reranking:
            hybrid_scores = self._rerank(query, hybrid_scores)

        # 상위 k개 선택
        sorted_tables = sorted(hybrid_scores.items(), key=lambda x: -x[1])
        selected = [t[0] for t in sorted_tables[:top_k]]

        return SchemaLinkingResult(
            query=query,
            selected_tables=selected,
            scores=hybrid_scores,
        )

    def _rerank(self, query: str, scores: dict[str, float]) -> dict[str, float]:
        """LLM 기반 재순위화 (Mock).
        
        실제 환경에서는 LLM을 호출하여 재순위화합니다.
        여기서는 기존 점수에 약간의 조정만 적용합니다.
        """
        # Mock: 질의에 포함된 키워드가 테이블명에 있으면 점수 부스트
        keywords = extract_keywords(query)
        
        reranked = {}
        for table, score in scores.items():
            boost = 0.0
            for kw in keywords:
                if kw.upper() in table.upper():
                    boost += 0.1
            reranked[table] = min(score + boost, 1.0)
        
        return reranked


# ============================================================================
# 테스트 실행
# ============================================================================


def run_tests() -> None:
    """테스트 케이스 실행."""
    print("=" * 80)
    print("하이브리드 검색 기반 스키마 링킹 테스트")
    print("=" * 80)
    
    # 스키마 DB 및 링커 초기화
    schema_db = MockSchemaDatabase()
    
    # 가중치 변화에 따른 성능 비교
    weight_configs = [
        (1.0, 0.0, "Vector Only"),
        (0.0, 1.0, "BM25 Only"),
        (0.7, 0.3, "Hybrid (0.7:0.3)"),
        (0.5, 0.5, "Hybrid (0.5:0.5)"),
    ]
    
    for vec_w, bm25_w, config_name in weight_configs:
        print(f"\n\n{'='*40}")
        print(f"설정: {config_name}")
        print(f"{'='*40}")
        
        linker = HybridSchemaLinker(
            schema_db=schema_db,
            vector_weight=vec_w,
            use_reranking=False,
        )
        
        results = []
        for test_case in TEST_CASES:
            result = linker.link(test_case["query"], top_k=5)
            result.ground_truth = test_case["ground_truth"]
            results.append(result)
        
        # 결과 출력
        print_result_table(results)
        
        # 집계 지표
        aggregate = compute_aggregate_metrics(results)
        print("\n📊 집계 지표:")
        for metric, value in aggregate.items():
            print(f"  {metric}: {value:.4f}")


def run_single_query(query: str, vector_weight: float = 0.7, use_reranking: bool = False) -> None:
    """단일 질의 테스트."""
    print(f"\n질의: {query}")
    print(f"가중치: Vector={vector_weight}, BM25={1-vector_weight}")
    print(f"Re-ranking: {'활성화' if use_reranking else '비활성화'}")
    print("-" * 60)
    
    schema_db = MockSchemaDatabase()
    linker = HybridSchemaLinker(
        schema_db=schema_db,
        vector_weight=vector_weight,
        use_reranking=use_reranking,
    )
    
    result = linker.link(query, top_k=5)
    
    print("\n선정된 테이블:")
    for i, table in enumerate(result.selected_tables, 1):
        score = result.scores.get(table, 0.0)
        print(f"  {i}. {table} (score: {score:.4f})")
    
    # 테이블 상세 정보
    print("\n테이블 상세:")
    for table_name in result.selected_tables[:3]:
        table = schema_db.get_table(table_name)
        if table:
            print(f"\n  📋 {table.name}")
            print(f"     설명: {table.description}")
            print(f"     컬럼: {', '.join(table.column_names[:5])}...")


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="하이브리드 검색 기반 스키마 링킹 테스트",
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
        "--vector-weight",
        type=float,
        default=0.7,
        help="벡터 검색 가중치 (기본값: 0.7)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="LLM 기반 재순위화 활성화",
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.query:
        run_single_query(args.query, args.vector_weight, args.rerank)
    else:
        # 기본: 간단한 데모
        print("하이브리드 검색 스키마 링킹 데모")
        print("-" * 40)
        run_single_query("M10 팹의 수율 데이터를 보여줘")
        run_single_query("설비별 생산량과 불량률")


if __name__ == "__main__":
    main()
