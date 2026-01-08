"""Pipeline 테스트."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from text2sql.core.models import RawSQLLog, NormalizedSQL, SQLTemplate
from text2sql.offline.pipeline import OfflinePipeline, PipelineResult


class TestOfflinePipeline:
    """OfflinePipeline 테스트."""

    def test_run_should_execute_all_stages(self) -> None:
        """15.1 전체 파이프라인이 모든 단계를 실행해야 함."""
        # Given
        mock_log_collector = MagicMock()
        mock_log_collector.collect.return_value = [
            RawSQLLog(
                sql_id="sql_001",
                sql_text="SELECT * FROM users WHERE id = 1",
                exec_count=100,
                error_count=0,
                collected_at=datetime(2026, 1, 8, 10, 0, 0),
            )
        ]

        mock_log_filter = MagicMock()
        mock_log_filter.filter.return_value = [
            RawSQLLog(
                sql_id="sql_001",
                sql_text="SELECT * FROM users WHERE id = 1",
                exec_count=100,
                error_count=0,
                collected_at=datetime(2026, 1, 8, 10, 0, 0),
            )
        ]

        mock_sql_normalizer = MagicMock()
        mock_sql_normalizer.normalize_literals.return_value = (
            "SELECT * FROM users WHERE id = :placeholder"
        )
        mock_sql_normalizer.extract_tables.return_value = ["users"]
        mock_sql_normalizer.extract_columns.return_value = ["*"]

        mock_description_generator = MagicMock()
        mock_description_generator.generate.return_value = "사용자 정보 조회 쿼리"

        mock_vector_indexer = MagicMock()
        mock_vector_indexer.index_batch.return_value = [1]

        mock_es_indexer = MagicMock()
        mock_es_indexer.index_batch.return_value = (1, [])

        pipeline = OfflinePipeline(
            log_collector=mock_log_collector,
            log_filter=mock_log_filter,
            sql_normalizer=mock_sql_normalizer,
            description_generator=mock_description_generator,
            vector_indexer=mock_vector_indexer,
            es_indexer=mock_es_indexer,
        )

        # When
        result = pipeline.run()

        # Then
        assert isinstance(result, PipelineResult)
        mock_log_collector.collect.assert_called_once()
        mock_log_filter.filter.assert_called_once()
        mock_description_generator.generate.assert_called()
        mock_vector_indexer.index_batch.assert_called_once()
        mock_es_indexer.index_batch.assert_called_once()

    def test_run_collect_filter_normalize_flow(self) -> None:
        """15.2 수집 → 필터링 → 정규화 플로우가 올바르게 동작해야 함."""
        # Given
        mock_log_collector = MagicMock()
        raw_logs = [
            RawSQLLog(
                sql_id="sql_001",
                sql_text="SELECT * FROM users WHERE id = 1",
                exec_count=100,
                error_count=0,
                collected_at=datetime(2026, 1, 8, 10, 0, 0),
            ),
            RawSQLLog(
                sql_id="sql_002",
                sql_text="SELECT * FROM orders WHERE user_id = 2",
                exec_count=50,
                error_count=0,
                collected_at=datetime(2026, 1, 8, 10, 0, 0),
            ),
        ]
        mock_log_collector.collect.return_value = raw_logs

        mock_log_filter = MagicMock()
        # 첫 번째 로그만 필터 통과 가정
        mock_log_filter.filter.return_value = [raw_logs[0]]

        mock_sql_normalizer = MagicMock()
        mock_sql_normalizer.normalize_literals.return_value = (
            "SELECT * FROM users WHERE id = :placeholder"
        )
        mock_sql_normalizer.extract_tables.return_value = ["users"]
        mock_sql_normalizer.extract_columns.return_value = ["*"]

        mock_description_generator = MagicMock()
        mock_description_generator.generate.return_value = "사용자 조회"

        mock_vector_indexer = MagicMock()
        mock_vector_indexer.index_batch.return_value = [1]

        mock_es_indexer = MagicMock()
        mock_es_indexer.index_batch.return_value = (1, [])

        pipeline = OfflinePipeline(
            log_collector=mock_log_collector,
            log_filter=mock_log_filter,
            sql_normalizer=mock_sql_normalizer,
            description_generator=mock_description_generator,
            vector_indexer=mock_vector_indexer,
            es_indexer=mock_es_indexer,
        )

        # When
        result = pipeline.run()

        # Then
        assert result.collected_count == 2  # 원래 수집된 로그 수
        assert result.filtered_count == 1  # 필터링 후 로그 수
        assert result.normalized_count == 1  # 정규화된 템플릿 수

        # 필터에 원본 로그가 전달되었는지 확인
        mock_log_filter.filter.assert_called_once_with(raw_logs)

    def test_run_normalize_describe_index_flow(self) -> None:
        """15.3 정규화 → 설명생성 → 인덱싱 플로우가 올바르게 동작해야 함."""
        # Given
        mock_log_collector = MagicMock()
        raw_log = RawSQLLog(
            sql_id="sql_001",
            sql_text="SELECT name, email FROM customers WHERE id = 1",
            exec_count=100,
            error_count=0,
            collected_at=datetime(2026, 1, 8, 10, 0, 0),
        )
        mock_log_collector.collect.return_value = [raw_log]

        mock_log_filter = MagicMock()
        mock_log_filter.filter.return_value = [raw_log]

        mock_sql_normalizer = MagicMock()
        normalized_sql = "SELECT name, email FROM customers WHERE id = :placeholder"
        mock_sql_normalizer.normalize_literals.return_value = normalized_sql
        mock_sql_normalizer.extract_tables.return_value = ["customers"]
        mock_sql_normalizer.extract_columns.return_value = ["name", "email"]

        mock_description_generator = MagicMock()
        description = "고객의 이름과 이메일을 조회하는 쿼리"
        mock_description_generator.generate.return_value = description

        mock_vector_indexer = MagicMock()
        mock_vector_indexer.index_batch.return_value = [1]

        mock_es_indexer = MagicMock()
        mock_es_indexer.index_batch.return_value = (1, [])

        pipeline = OfflinePipeline(
            log_collector=mock_log_collector,
            log_filter=mock_log_filter,
            sql_normalizer=mock_sql_normalizer,
            description_generator=mock_description_generator,
            vector_indexer=mock_vector_indexer,
            es_indexer=mock_es_indexer,
        )

        # When
        result = pipeline.run()

        # Then
        # 정규화기가 원본 SQL로 호출되었는지
        mock_sql_normalizer.normalize_literals.assert_called_with(raw_log.sql_text)

        # 설명 생성기가 정규화된 SQL로 호출되었는지
        mock_description_generator.generate.assert_called_with(normalized_sql)

        # 인덱서에 전달된 템플릿 검증
        vector_call_args = mock_vector_indexer.index_batch.call_args
        templates = vector_call_args[0][0]
        assert len(templates) == 1
        assert templates[0].template_text == normalized_sql
        assert templates[0].description == description
        assert templates[0].tables == ["customers"]
        assert templates[0].columns == ["name", "email"]
        assert templates[0].exec_count == 100

        # ES 인덱서도 동일한 템플릿 수신
        es_call_args = mock_es_indexer.index_batch.call_args
        es_templates = es_call_args[0][0]
        assert len(es_templates) == 1

    def test_run_should_capture_errors_and_continue(self) -> None:
        """15.4 에러 발생 시 에러를 캡처하고 계속 진행해야 함."""
        # Given
        mock_log_collector = MagicMock()
        mock_log_collector.collect.return_value = [
            RawSQLLog(
                sql_id="sql_001",
                sql_text="SELECT * FROM users",
                exec_count=100,
                error_count=0,
                collected_at=datetime(2026, 1, 8, 10, 0, 0),
            ),
            RawSQLLog(
                sql_id="sql_002",
                sql_text="SELECT * FROM orders",
                exec_count=50,
                error_count=0,
                collected_at=datetime(2026, 1, 8, 10, 0, 0),
            ),
        ]

        mock_log_filter = MagicMock()
        mock_log_filter.filter.return_value = mock_log_collector.collect.return_value

        mock_sql_normalizer = MagicMock()
        mock_sql_normalizer.normalize_literals.return_value = "SELECT * FROM users"
        mock_sql_normalizer.extract_tables.return_value = ["users"]
        mock_sql_normalizer.extract_columns.return_value = ["*"]

        mock_description_generator = MagicMock()
        # 첫 번째 호출은 성공, 두 번째 호출은 에러 발생
        mock_description_generator.generate.side_effect = [
            "사용자 조회",
            Exception("LLM API 에러"),
        ]

        mock_vector_indexer = MagicMock()
        mock_vector_indexer.index_batch.return_value = [1]

        mock_es_indexer = MagicMock()
        mock_es_indexer.index_batch.return_value = (1, [])

        pipeline = OfflinePipeline(
            log_collector=mock_log_collector,
            log_filter=mock_log_filter,
            sql_normalizer=mock_sql_normalizer,
            description_generator=mock_description_generator,
            vector_indexer=mock_vector_indexer,
            es_indexer=mock_es_indexer,
        )

        # When
        result = pipeline.run()

        # Then
        # 에러가 발생해도 성공한 템플릿은 인덱싱되어야 함
        assert result.normalized_count >= 1
        # 에러가 기록되어야 함
        assert len(result.errors) >= 1
        assert result.success is False

    def test_result_should_contain_summary_report(self) -> None:
        """15.5 파이프라인 결과에 요약 리포트가 포함되어야 함."""
        # Given
        mock_log_collector = MagicMock()
        mock_log_collector.collect.return_value = [
            RawSQLLog(
                sql_id="sql_001",
                sql_text="SELECT * FROM users",
                exec_count=100,
                error_count=0,
                collected_at=datetime(2026, 1, 8, 10, 0, 0),
            ),
            RawSQLLog(
                sql_id="sql_002",
                sql_text="SELECT * FROM orders",
                exec_count=50,
                error_count=0,
                collected_at=datetime(2026, 1, 8, 10, 0, 0),
            ),
        ]

        mock_log_filter = MagicMock()
        mock_log_filter.filter.return_value = mock_log_collector.collect.return_value

        mock_sql_normalizer = MagicMock()
        mock_sql_normalizer.normalize_literals.return_value = "SELECT * FROM users"
        mock_sql_normalizer.extract_tables.return_value = ["users"]
        mock_sql_normalizer.extract_columns.return_value = ["*"]

        mock_description_generator = MagicMock()
        mock_description_generator.generate.return_value = "테스트 설명"

        mock_vector_indexer = MagicMock()
        mock_vector_indexer.index_batch.return_value = [1, 2]

        mock_es_indexer = MagicMock()
        mock_es_indexer.index_batch.return_value = (2, [])

        pipeline = OfflinePipeline(
            log_collector=mock_log_collector,
            log_filter=mock_log_filter,
            sql_normalizer=mock_sql_normalizer,
            description_generator=mock_description_generator,
            vector_indexer=mock_vector_indexer,
            es_indexer=mock_es_indexer,
        )

        # When
        result = pipeline.run()

        # Then
        # 결과에 리포트 데이터가 포함되어야 함
        assert result.collected_count == 2
        assert result.filtered_count == 2
        assert result.normalized_count == 2
        assert result.indexed_count == 2
        assert result.errors == []
        assert result.success is True

        # 리포트 문자열 생성 가능해야 함
        report = result.to_report()
        assert "수집된 로그" in report
        assert "필터링된 로그" in report
        assert "정규화된 템플릿" in report
        assert "인덱싱된 템플릿" in report

