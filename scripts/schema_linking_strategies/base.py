#!/usr/bin/env python
"""스키마 링킹 테스트를 위한 공통 기반 모듈.

공용 데이터 클래스, 샘플 스키마 데이터, 평가 지표 계산 유틸리티를 제공합니다.
"""

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================================
# 데이터 클래스
# ============================================================================


@dataclass
class ColumnInfo:
    """컬럼 정보."""

    name: str
    data_type: str
    description: str = ""
    is_primary_key: bool = False
    is_foreign_key: bool = False
    sample_values: list[str] = field(default_factory=list)


@dataclass
class TableInfo:
    """테이블 정보."""

    name: str
    description: str
    columns: list[ColumnInfo] = field(default_factory=list)
    purpose: str = ""  # 테이블 목적 (LLM 필터링용)
    
    @property
    def column_names(self) -> list[str]:
        """컬럼명 리스트 반환."""
        return [col.name for col in self.columns]


@dataclass
class SchemaLinkingResult:
    """스키마 링킹 결과."""

    query: str
    selected_tables: list[str]
    scores: dict[str, float]  # table_name -> score
    ground_truth: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환."""
        return {
            "query": self.query,
            "selected_tables": self.selected_tables,
            "scores": self.scores,
            "ground_truth": self.ground_truth,
        }


@dataclass
class GlossaryTerm:
    """용어 사전 항목."""
    
    term: str
    korean_name: str
    description: str
    category: str = ""


# ============================================================================
# 샘플 스키마 데이터베이스
# ============================================================================


class MockSchemaDatabase:
    """테스트용 샘플 스키마 데이터."""

    def __init__(self):
        self.tables: dict[str, TableInfo] = {}
        self.glossary: list[GlossaryTerm] = []
        self._init_sample_tables()

    def _init_sample_tables(self):
        """MES 관련 샘플 테이블 초기화."""
        
        # --- MES_PRD_YIELD_M10 (생산 수율 테이블) ---
        self.tables["MES_PRD_YIELD_M10"] = TableInfo(
            name="MES_PRD_YIELD_M10",
            description="M10 팹의 생산 수율 데이터",
            purpose="M10 팹의 로트/웨이퍼별 수율을 저장하는 테이블. 품질 분석에 사용됨.",
            columns=[
                ColumnInfo("lot_id", "VARCHAR2(20)", "로트 식별자", is_primary_key=True, 
                          sample_values=["LOT20260108001", "LOT20260108002"]),
                ColumnInfo("wafer_id", "VARCHAR2(10)", "웨이퍼 식별자",
                          sample_values=["WFR001", "WFR002", "WFR003"]),
                ColumnInfo("yield_rate", "NUMBER(5,2)", "수율 (%)",
                          sample_values=["95.5", "97.2", "92.8"]),
                ColumnInfo("defect_cnt", "NUMBER", "불량 수",
                          sample_values=["12", "5", "23"]),
                ColumnInfo("create_dt", "DATE", "생성 일시"),
                ColumnInfo("fab_id", "VARCHAR2(10)", "팹 식별자", is_foreign_key=True,
                          sample_values=["M10"]),
            ],
        )

        # --- MES_PRD_YIELD_M11 (생산 수율 테이블) ---
        self.tables["MES_PRD_YIELD_M11"] = TableInfo(
            name="MES_PRD_YIELD_M11",
            description="M11 팹의 생산 수율 데이터",
            purpose="M11 팹의 로트/웨이퍼별 수율 저장 테이블.",
            columns=[
                ColumnInfo("lot_id", "VARCHAR2(20)", "로트 식별자", is_primary_key=True),
                ColumnInfo("wafer_id", "VARCHAR2(10)", "웨이퍼 식별자"),
                ColumnInfo("yield_rate", "NUMBER(5,2)", "수율 (%)"),
                ColumnInfo("defect_cnt", "NUMBER", "불량 수"),
                ColumnInfo("create_dt", "DATE", "생성 일시"),
                ColumnInfo("fab_id", "VARCHAR2(10)", "팹 식별자", is_foreign_key=True),
                ColumnInfo("product_id", "VARCHAR2(20)", "제품 식별자"),
            ],
        )

        # --- MES_EQP_MST_M10 (설비 마스터) ---
        self.tables["MES_EQP_MST_M10"] = TableInfo(
            name="MES_EQP_MST_M10",
            description="M10 팹의 설비 마스터 정보",
            purpose="M10 팹 내 모든 설비(장비)의 기본 정보를 관리하는 마스터 테이블.",
            columns=[
                ColumnInfo("eqp_id", "VARCHAR2(20)", "설비 식별자", is_primary_key=True,
                          sample_values=["EQP_PHOTO_01", "EQP_ETCH_01"]),
                ColumnInfo("eqp_name", "VARCHAR2(100)", "설비명",
                          sample_values=["ASML Photo 1", "LAM Etch 1"]),
                ColumnInfo("eqp_type", "VARCHAR2(20)", "설비 유형",
                          sample_values=["PHOTO", "ETCH", "DEPO"]),
                ColumnInfo("chamber_cnt", "NUMBER", "챔버 수"),
                ColumnInfo("status", "VARCHAR2(10)", "상태",
                          sample_values=["RUN", "DOWN", "IDLE"]),
            ],
        )

        # --- MES_BIZ_LOTHOLD_INF_M10 (로트 홀드 정보) ---
        self.tables["MES_BIZ_LOTHOLD_INF_M10"] = TableInfo(
            name="MES_BIZ_LOTHOLD_INF_M10",
            description="M10 팹의 로트 홀드 정보",
            purpose="로트가 공정 중단(홀드)된 이력을 관리하는 테이블.",
            columns=[
                ColumnInfo("lot_id", "VARCHAR2(20)", "로트 식별자", is_foreign_key=True),
                ColumnInfo("hold_code", "VARCHAR2(10)", "홀드 코드"),
                ColumnInfo("hold_reason", "VARCHAR2(200)", "홀드 사유"),
                ColumnInfo("hold_dt", "DATE", "홀드 일시"),
                ColumnInfo("release_dt", "DATE", "홀드 해제 일시"),
            ],
        )

        # --- MES_TRK_HIS_M10 (트래킹 이력) ---
        self.tables["MES_TRK_HIS_M10"] = TableInfo(
            name="MES_TRK_HIS_M10",
            description="M10 팹의 로트 트래킹 이력",
            purpose="로트가 각 공정(설비)을 거치는 이력을 저장.",
            columns=[
                ColumnInfo("lot_id", "VARCHAR2(20)", "로트 식별자", is_foreign_key=True),
                ColumnInfo("wafer_id", "VARCHAR2(10)", "웨이퍼 식별자"),
                ColumnInfo("eqp_id", "VARCHAR2(20)", "설비 식별자", is_foreign_key=True),
                ColumnInfo("proc_id", "VARCHAR2(20)", "공정 식별자"),
                ColumnInfo("recipe_id", "VARCHAR2(20)", "레시피 식별자"),
                ColumnInfo("track_in_dt", "DATE", "트랙인 일시"),
                ColumnInfo("track_out_dt", "DATE", "트랙아웃 일시"),
                ColumnInfo("cycle_time", "NUMBER", "사이클 타임 (분)"),
            ],
        )

        # --- MES_PROC_MST_M10 (공정 마스터) ---
        self.tables["MES_PROC_MST_M10"] = TableInfo(
            name="MES_PROC_MST_M10",
            description="M10 팹의 공정 마스터 정보",
            purpose="반도체 제조 공정 정의 및 순서를 관리하는 마스터 테이블.",
            columns=[
                ColumnInfo("proc_id", "VARCHAR2(20)", "공정 식별자", is_primary_key=True),
                ColumnInfo("proc_name", "VARCHAR2(100)", "공정명",
                          sample_values=["Photo Litho", "Etch", "Deposition"]),
                ColumnInfo("proc_seq", "NUMBER", "공정 순서"),
                ColumnInfo("proc_type", "VARCHAR2(20)", "공정 유형"),
            ],
        )

        # --- MES_DEF_HIS_M10 (불량 이력) ---
        self.tables["MES_DEF_HIS_M10"] = TableInfo(
            name="MES_DEF_HIS_M10",
            description="M10 팹의 불량 이력",
            purpose="제품 불량 발생 이력을 기록하는 테이블.",
            columns=[
                ColumnInfo("lot_id", "VARCHAR2(20)", "로트 식별자", is_foreign_key=True),
                ColumnInfo("wafer_id", "VARCHAR2(10)", "웨이퍼 식별자"),
                ColumnInfo("defect_code", "VARCHAR2(20)", "불량 코드"),
                ColumnInfo("defect_cnt", "NUMBER", "불량 수"),
                ColumnInfo("create_dt", "DATE", "발생 일시"),
            ],
        )
        
        # --- MES_RCP_MST_M10 (레시피 마스터) ---
        self.tables["MES_RCP_MST_M10"] = TableInfo(
            name="MES_RCP_MST_M10",
            description="M10 팹의 레시피 마스터 정보",
            purpose="설비 운영을 위한 레시피 파라미터 관리 테이블.",
            columns=[
                ColumnInfo("recipe_id", "VARCHAR2(20)", "레시피 식별자", is_primary_key=True),
                ColumnInfo("recipe_name", "VARCHAR2(100)", "레시피명"),
                ColumnInfo("eqp_id", "VARCHAR2(20)", "설비 식별자", is_foreign_key=True),
                ColumnInfo("version", "NUMBER", "버전"),
            ],
        )
        
        # --- MES_WIP_HIS_M10 (재공 이력) ---
        self.tables["MES_WIP_HIS_M10"] = TableInfo(
            name="MES_WIP_HIS_M10",
            description="M10 팹의 재공(WIP) 이력",
            purpose="공정 중인 재공 현황을 추적하는 테이블.",
            columns=[
                ColumnInfo("lot_id", "VARCHAR2(20)", "로트 식별자"),
                ColumnInfo("proc_id", "VARCHAR2(20)", "공정 식별자"),
                ColumnInfo("step_id", "VARCHAR2(20)", "스텝 식별자"),
                ColumnInfo("wip_qty", "NUMBER", "재공 수량"),
                ColumnInfo("queue_time", "NUMBER", "대기 시간 (분)"),
                ColumnInfo("create_dt", "DATE", "생성 일시"),
            ],
        )

    def get_all_tables(self) -> list[TableInfo]:
        """모든 테이블 정보 반환."""
        return list(self.tables.values())

    def get_table(self, name: str) -> Optional[TableInfo]:
        """테이블명으로 테이블 정보 조회."""
        return self.tables.get(name)
    
    def get_table_names(self) -> list[str]:
        """모든 테이블명 반환."""
        return list(self.tables.keys())

    def load_glossary(self, glossary_path: Path) -> None:
        """CSV 용어 사전 로드."""
        if not glossary_path.exists():
            return
        
        with open(glossary_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                term = GlossaryTerm(
                    term=row.get("term", ""),
                    korean_name=row.get("korean_name", ""),
                    description=row.get("description", ""),
                    category=row.get("category", ""),
                )
                self.glossary.append(term)
    
    def find_synonyms(self, word: str) -> list[str]:
        """용어 사전에서 동의어 찾기."""
        synonyms = []
        word_lower = word.lower()
        for term in self.glossary:
            if (word_lower in term.term.lower() or 
                word_lower in term.korean_name or
                word_lower in term.description):
                synonyms.append(term.term)
                if term.korean_name:
                    synonyms.append(term.korean_name)
        return list(set(synonyms))


# ============================================================================
# 평가 지표 계산
# ============================================================================


class EvaluationMetrics:
    """스키마 링킹 평가 지표 계산."""

    @staticmethod
    def hit_rate_at_k(predicted: list[str], ground_truth: list[str], k: int = 5) -> float:
        """Hit Rate@K: 상위 k개 예측에 정답이 포함된 비율.
        
        Args:
            predicted: 예측된 테이블 리스트 (순위순)
            ground_truth: 정답 테이블 리스트
            k: 상위 k개까지 검사
            
        Returns:
            1.0 if any ground_truth in top-k, else 0.0
        """
        if not ground_truth:
            return 0.0
        
        top_k = set(predicted[:k])
        gt_set = set(ground_truth)
        
        return 1.0 if top_k & gt_set else 0.0

    @staticmethod
    def mrr(predicted: list[str], ground_truth: list[str]) -> float:
        """Mean Reciprocal Rank: 첫 번째 정답의 역순위 평균.
        
        Args:
            predicted: 예측된 테이블 리스트 (순위순)
            ground_truth: 정답 테이블 리스트
            
        Returns:
            1/rank of first correct prediction, 0 if none found
        """
        if not ground_truth:
            return 0.0
        
        gt_set = set(ground_truth)
        for i, table in enumerate(predicted, 1):
            if table in gt_set:
                return 1.0 / i
        return 0.0

    @staticmethod
    def precision(predicted: list[str], ground_truth: list[str]) -> float:
        """Precision: 예측 중 정답 비율."""
        if not predicted:
            return 0.0
        
        correct = len(set(predicted) & set(ground_truth))
        return correct / len(predicted)

    @staticmethod
    def recall(predicted: list[str], ground_truth: list[str]) -> float:
        """Recall: 정답 중 예측된 비율."""
        if not ground_truth:
            return 0.0
        
        correct = len(set(predicted) & set(ground_truth))
        return correct / len(ground_truth)

    @staticmethod
    def f1_score(predicted: list[str], ground_truth: list[str]) -> float:
        """F1 Score: Precision과 Recall의 조화평균."""
        p = EvaluationMetrics.precision(predicted, ground_truth)
        r = EvaluationMetrics.recall(predicted, ground_truth)
        
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def evaluate_all(predicted: list[str], ground_truth: list[str], k: int = 5) -> dict:
        """모든 지표 계산."""
        return {
            "hit_rate_at_k": EvaluationMetrics.hit_rate_at_k(predicted, ground_truth, k),
            "mrr": EvaluationMetrics.mrr(predicted, ground_truth),
            "precision": EvaluationMetrics.precision(predicted, ground_truth),
            "recall": EvaluationMetrics.recall(predicted, ground_truth),
            "f1_score": EvaluationMetrics.f1_score(predicted, ground_truth),
        }


# ============================================================================
# 테스트 케이스 정의
# ============================================================================


# 테스트용 질의-정답 쌍
TEST_CASES = [
    {
        "query": "M10 팹의 수율 데이터를 보여줘",
        "ground_truth": ["MES_PRD_YIELD_M10"],
        "description": "단일 테이블 - 수율 조회",
    },
    {
        "query": "M10 팹 설비별 생산량과 불량률을 분석해줘",
        "ground_truth": ["MES_EQP_MST_M10", "MES_TRK_HIS_M10", "MES_DEF_HIS_M10"],
        "description": "다중 테이블 JOIN - 설비/생산/불량",
    },
    {
        "query": "로트 홀드 이력과 사유를 조회해줘",
        "ground_truth": ["MES_BIZ_LOTHOLD_INF_M10"],
        "description": "단일 테이블 - 홀드 이력",
    },
    {
        "query": "공정별 사이클 타임과 레시피 정보가 필요해",
        "ground_truth": ["MES_TRK_HIS_M10", "MES_PROC_MST_M10", "MES_RCP_MST_M10"],
        "description": "다중 테이블 JOIN - 공정/트래킹/레시피",
    },
    {
        "query": "재공 현황과 대기 시간을 확인하고 싶어",
        "ground_truth": ["MES_WIP_HIS_M10"],
        "description": "단일 테이블 - WIP",
    },
    {
        "query": "웨이퍼별 defect 현황을 분석해줘",
        "ground_truth": ["MES_DEF_HIS_M10"],
        "description": "단일 테이블 - 불량 (영어 용어)",
    },
    {
        "query": "Photo 공정 설비의 가동 상태를 보여줘",
        "ground_truth": ["MES_EQP_MST_M10"],
        "description": "단일 테이블 - 설비 상태 (공정 유형 필터)",
    },
]


# ============================================================================
# 유틸리티 함수
# ============================================================================


def extract_keywords(text: str) -> list[str]:
    """텍스트에서 키워드 추출 (간단한 토큰화)."""
    # 한글, 영문, 숫자 추출
    words = re.findall(r"[가-힣]+|[A-Za-z0-9_]+", text)
    # 불용어 제거
    stopwords = {"을", "를", "이", "가", "에", "의", "로", "와", "과", "해", "줘", "해줘", "보여줘", "필요해"}
    return [w for w in words if w not in stopwords and len(w) > 1]


def print_result_table(results: list[SchemaLinkingResult]) -> None:
    """결과 테이블 출력."""
    print("\n" + "=" * 80)
    print("스키마 링킹 결과")
    print("=" * 80)
    
    for result in results:
        print(f"\n질의: {result.query}")
        print(f"정답 테이블: {result.ground_truth}")
        print(f"선정된 테이블: {result.selected_tables}")
        
        # 점수 출력
        print("점수:")
        for table, score in sorted(result.scores.items(), key=lambda x: -x[1])[:5]:
            marker = "✓" if table in result.ground_truth else " "
            print(f"  {marker} {table}: {score:.4f}")
        
        # 평가 지표
        metrics = EvaluationMetrics.evaluate_all(result.selected_tables, result.ground_truth)
        print(f"평가: Hit@5={metrics['hit_rate_at_k']:.2f}, MRR={metrics['mrr']:.2f}, "
              f"F1={metrics['f1_score']:.2f}")
    
    print("\n" + "=" * 80)


def compute_aggregate_metrics(results: list[SchemaLinkingResult]) -> dict:
    """여러 결과의 평균 지표 계산."""
    all_metrics = [
        EvaluationMetrics.evaluate_all(r.selected_tables, r.ground_truth)
        for r in results
    ]
    
    if not all_metrics:
        return {}
    
    aggregate = {}
    for key in all_metrics[0].keys():
        aggregate[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return aggregate


if __name__ == "__main__":
    # 기본 동작 테스트
    print("=== 스키마 링킹 기반 모듈 테스트 ===\n")
    
    # 샘플 스키마 로드
    db = MockSchemaDatabase()
    print(f"로드된 테이블 수: {len(db.tables)}")
    
    for table in db.get_all_tables():
        print(f"\n테이블: {table.name}")
        print(f"  설명: {table.description}")
        print(f"  컬럼: {table.column_names}")
    
    # 평가 지표 테스트
    print("\n\n=== 평가 지표 테스트 ===")
    predicted = ["MES_PRD_YIELD_M10", "MES_EQP_MST_M10", "MES_TRK_HIS_M10"]
    ground_truth = ["MES_PRD_YIELD_M10", "MES_DEF_HIS_M10"]
    
    metrics = EvaluationMetrics.evaluate_all(predicted, ground_truth)
    print(f"예측: {predicted}")
    print(f"정답: {ground_truth}")
    print(f"지표: {metrics}")
