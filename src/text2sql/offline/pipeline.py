"""오프라인 파이프라인 오케스트레이터."""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from text2sql.core.models import RawSQLLog, NormalizedSQL, SQLTemplate


class PipelineStage(Enum):
    """파이프라인 단계."""

    COLLECTING = "collecting"
    FILTERING = "filtering"
    NORMALIZING = "normalizing"
    GENERATING_DESC = "generating_desc"
    INDEXING_VECTOR = "indexing_vector"
    INDEXING_ES = "indexing_es"
    COMPLETED = "completed"


@dataclass
class ProgressInfo:
    """진행 상황 정보."""

    stage: PipelineStage
    current: int = 0
    total: int = 0
    message: str = ""
    sql_id: str = ""


# 진행 상황 콜백 타입
ProgressCallback = Callable[[ProgressInfo], None]


@dataclass
class PipelineResult:
    """파이프라인 실행 결과."""

    collected_count: int = 0
    filtered_count: int = 0
    normalized_count: int = 0
    indexed_count: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """파이프라인 성공 여부."""
        return len(self.errors) == 0

    def to_report(self) -> str:
        """파이프라인 결과를 리포트 문자열로 변환.

        Returns:
            리포트 문자열
        """
        lines = [
            "=== 파이프라인 실행 결과 ===",
            f"수집된 로그: {self.collected_count}건",
            f"필터링된 로그: {self.filtered_count}건",
            f"정규화된 템플릿: {self.normalized_count}건",
            f"인덱싱된 템플릿: {self.indexed_count}건",
            f"에러 수: {len(self.errors)}건",
            f"성공 여부: {'성공' if self.success else '실패'}",
        ]

        if self.errors:
            lines.append("\n=== 에러 목록 ===")
            for error in self.errors:
                lines.append(
                    f"  - {error.get('sql_id', 'N/A')}: {error.get('error', 'Unknown')}"
                )

        return "\n".join(lines)


class OfflinePipeline:
    """오프라인 파이프라인 오케스트레이터."""

    def __init__(
        self,
        log_collector: Any,
        log_filter: Any,
        sql_normalizer: Any,
        description_generator: Any,
        vector_indexer: Any,
        es_indexer: Any,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """파이프라인 초기화.

        Args:
            log_collector: 로그 수집기
            log_filter: 로그 필터
            sql_normalizer: SQL 정규화기
            description_generator: 설명 생성기
            vector_indexer: 벡터 인덱서
            es_indexer: ES 인덱서
            progress_callback: 진행 상황 콜백 함수
        """
        self._log_collector = log_collector
        self._log_filter = log_filter
        self._sql_normalizer = sql_normalizer
        self._description_generator = description_generator
        self._vector_indexer = vector_indexer
        self._es_indexer = es_indexer
        self._progress_callback = progress_callback

    def _notify_progress(self, info: ProgressInfo) -> None:
        """진행 상황을 알림."""
        if self._progress_callback:
            self._progress_callback(info)

    def run(self) -> PipelineResult:
        """파이프라인을 실행.

        Returns:
            파이프라인 실행 결과
        """
        result = PipelineResult()

        # 1. 로그 수집
        self._notify_progress(ProgressInfo(
            stage=PipelineStage.COLLECTING,
            message="Oracle에서 SQL 로그 수집 중..."
        ))
        raw_logs = self._log_collector.collect()
        result.collected_count = len(raw_logs)

        # 2. 로그 필터링
        self._notify_progress(ProgressInfo(
            stage=PipelineStage.FILTERING,
            total=len(raw_logs),
            message="SQL 로그 필터링 중..."
        ))
        filtered_logs = self._log_filter.filter(raw_logs)
        result.filtered_count = len(filtered_logs)

        # 3. 정규화 및 설명 생성
        templates = self._process_logs(filtered_logs, result)
        result.normalized_count = len(templates)

        # 4. 벡터 인덱싱
        if templates:
            self._notify_progress(ProgressInfo(
                stage=PipelineStage.INDEXING_VECTOR,
                total=len(templates),
                message="Milvus 벡터 인덱싱 중..."
            ))
            self._vector_indexer.index_batch(templates)

            # 5. ES 인덱싱
            self._notify_progress(ProgressInfo(
                stage=PipelineStage.INDEXING_ES,
                total=len(templates),
                message="Elasticsearch 인덱싱 중..."
            ))
            self._es_indexer.index_batch(templates)
            result.indexed_count = len(templates)

        # 완료
        self._notify_progress(ProgressInfo(
            stage=PipelineStage.COMPLETED,
            message="파이프라인 완료!"
        ))

        return result

    def _process_logs(
        self, logs: list[RawSQLLog], result: PipelineResult
    ) -> list[SQLTemplate]:
        """로그를 정규화하고 템플릿으로 변환.

        Args:
            logs: 원본 로그 리스트
            result: 파이프라인 결과 객체 (에러 기록용)

        Returns:
            SQL 템플릿 리스트
        """
        templates = []
        total = len(logs)

        for idx, log in enumerate(logs):
            try:
                # 정규화 진행 상황
                self._notify_progress(ProgressInfo(
                    stage=PipelineStage.NORMALIZING,
                    current=idx + 1,
                    total=total,
                    message=f"SQL 정규화 중...",
                    sql_id=log.sql_id,
                ))

                normalized_text = self._sql_normalizer.normalize_literals(log.sql_text)
                tables = self._sql_normalizer.extract_tables(log.sql_text)
                columns = self._sql_normalizer.extract_columns(log.sql_text)

                # 템플릿 해시 생성
                template_hash = hashlib.md5(normalized_text.encode()).hexdigest()

                # 설명 생성 진행 상황
                self._notify_progress(ProgressInfo(
                    stage=PipelineStage.GENERATING_DESC,
                    current=idx + 1,
                    total=total,
                    message=f"LLM 설명 생성 중...",
                    sql_id=log.sql_id,
                ))

                description = self._description_generator.generate(normalized_text)

                template = SQLTemplate(
                    template_id=template_hash,
                    template_text=normalized_text,
                    description=description,
                    tables=tables,
                    columns=columns,
                    exec_count=log.exec_count,
                )
                templates.append(template)
            except Exception as e:
                result.errors.append(
                    {
                        "sql_id": log.sql_id,
                        "stage": "processing",
                        "error": str(e),
                    }
                )

        return templates

