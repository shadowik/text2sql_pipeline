#!/usr/bin/env python
"""오프라인 파이프라인 실행 스크립트.

사용법:
    python scripts/run_pipeline.py                    # 기본 실행
    python scripts/run_pipeline.py --limit 100        # 상위 100개 로그만 처리
    python scripts/run_pipeline.py --dry-run          # 실제 저장 없이 테스트
    python scripts/run_pipeline.py --demo             # 샘플 데이터로 데모 실행
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.layout import Layout
from rich.text import Text

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text2sql.core.config import Settings
from text2sql.core.models import RawSQLLog
from text2sql.offline.ingestor.log_filter import LogFilter
from text2sql.offline.processor.sql_normalizer import SQLNormalizer
from text2sql.offline.pipeline import OfflinePipeline, PipelineResult, PipelineStage, ProgressInfo

console = Console()


class PipelineProgressUI:
    """파이프라인 진행 상황 UI."""

    STAGE_NAMES = {
        PipelineStage.COLLECTING: "📥 로그 수집",
        PipelineStage.FILTERING: "🔍 로그 필터링",
        PipelineStage.NORMALIZING: "⚙️  SQL 정규화",
        PipelineStage.GENERATING_DESC: "🤖 LLM 설명 생성",
        PipelineStage.INDEXING_VECTOR: "🔢 벡터 인덱싱",
        PipelineStage.INDEXING_ES: "🔎 ES 인덱싱",
        PipelineStage.COMPLETED: "✅ 완료",
    }

    def __init__(self):
        self.current_stage = None
        self.current_sql_id = ""
        self.progress_current = 0
        self.progress_total = 0
        self.message = ""
        self.start_time = time.time()
        self.stage_times: dict[str, float] = {}
        self._last_stage_start = time.time()

    def update(self, info: ProgressInfo) -> None:
        """진행 상황 업데이트."""
        # 단계가 변경되면 이전 단계 시간 기록
        if self.current_stage != info.stage:
            if self.current_stage:
                self.stage_times[self.current_stage.value] = time.time() - self._last_stage_start
            self._last_stage_start = time.time()

        self.current_stage = info.stage
        self.current_sql_id = info.sql_id
        self.progress_current = info.current
        self.progress_total = info.total
        self.message = info.message

    def generate_display(self) -> Panel:
        """화면 표시 생성."""
        layout = Layout()

        # 헤더
        elapsed = time.time() - self.start_time
        header = Text()
        header.append("🚀 Text2SQL 오프라인 파이프라인\n", style="bold cyan")
        header.append(f"⏱️  경과 시간: {elapsed:.1f}초", style="dim")

        # 단계 상태 테이블
        stage_table = Table(show_header=True, header_style="bold magenta", box=None)
        stage_table.add_column("단계", width=20)
        stage_table.add_column("상태", width=12)
        stage_table.add_column("소요 시간", width=10)

        for stage in PipelineStage:
            if stage == PipelineStage.COMPLETED:
                continue  # 완료 단계는 별도로 표시하지 않음

            name = self.STAGE_NAMES.get(stage, stage.value)
            if self.current_stage == stage:
                status = "🔄 진행중"
                style = "yellow"
                elapsed_stage = time.time() - self._last_stage_start
                time_str = f"{elapsed_stage:.1f}s"
            elif stage.value in self.stage_times:
                status = "✅ 완료"
                style = "green"
                time_str = f"{self.stage_times[stage.value]:.1f}s"
            elif self.current_stage == PipelineStage.COMPLETED:
                # 완료 상태인데 기록되지 않은 단계는 건너뛴 것으로 처리
                status = "⏭️ 건너뜀"
                style = "dim"
                time_str = "-"
            else:
                status = "⏳ 대기"
                style = "dim"
                time_str = "-"

            stage_table.add_row(name, status, time_str, style=style)

        # 진행 바
        progress_section = Text()
        if self.progress_total > 0:
            pct = (self.progress_current / self.progress_total) * 100
            bar_width = 30
            filled = int(bar_width * self.progress_current / self.progress_total)
            bar = "█" * filled + "░" * (bar_width - filled)
            progress_section.append(f"\n📊 진행률: [{bar}] {pct:.1f}%\n", style="cyan")
            progress_section.append(f"   {self.progress_current}/{self.progress_total} 처리됨\n")

        # 현재 작업
        current_work = Text()
        if self.current_sql_id:
            current_work.append(f"\n🔧 현재 처리 중: ", style="bold")
            current_work.append(f"{self.current_sql_id}\n", style="yellow")
        if self.message:
            current_work.append(f"   {self.message}", style="dim")

        # 전체 레이아웃 조합
        content = Text()
        content.append_text(header)
        content.append("\n\n")
        
        # 테이블을 문자열로 변환
        from io import StringIO
        from rich.console import Console as RichConsole
        str_io = StringIO()
        temp_console = RichConsole(file=str_io, force_terminal=True, width=60)
        temp_console.print(stage_table)
        content.append(str_io.getvalue())
        
        content.append_text(progress_section)
        content.append_text(current_work)

        return Panel(content, title="[bold blue]파이프라인 진행 상황[/bold blue]", border_style="blue")


class MockOracleAdapter:
    """Oracle 어댑터 Mock (데모용)."""

    def __init__(self, sample_logs: list[dict]):
        self._logs = sample_logs

    def execute_query(self, query: str) -> list[dict]:
        """샘플 데이터를 반환."""
        return self._logs


class MockLLMClient:
    """LLM 클라이언트 Mock (데모용)."""

    def invoke(self, message: str) -> str:
        """SQL에 대한 간단한 Mock 설명 생성."""
        time.sleep(0.1)  # 시뮬레이션을 위한 지연
        if "COUNT" in message or "SUM" in message:
            return "이 쿼리는 집계 함수를 사용하여 데이터를 요약합니다."
        if "JOIN" in message:
            return "이 쿼리는 여러 테이블을 조인하여 관련 데이터를 조회합니다."
        if "WHERE" in message:
            return "이 쿼리는 조건에 맞는 데이터를 필터링하여 조회합니다."
        return "이 쿼리는 데이터베이스에서 데이터를 조회합니다."


class MockVectorIndexer:
    """벡터 인덱서 Mock (데모/dry-run용)."""

    def __init__(self, dry_run: bool = False):
        self._dry_run = dry_run
        self._indexed = []

    def index_batch(self, templates):
        """템플릿을 인덱싱 (dry-run이면 저장하지 않음)."""
        time.sleep(0.3)  # 시뮬레이션을 위한 지연
        if not self._dry_run:
            self._indexed.extend(templates)


class MockESIndexer:
    """ES 인덱서 Mock (데모/dry-run용)."""

    def __init__(self, dry_run: bool = False):
        self._dry_run = dry_run
        self._indexed = []

    def index_batch(self, templates):
        """템플릿을 인덱싱 (dry-run이면 저장하지 않음)."""
        time.sleep(0.3)  # 시뮬레이션을 위한 지연
        if not self._dry_run:
            self._indexed.extend(templates)


class MockLogCollector:
    """로그 수집기 Mock (데모용)."""

    def __init__(self, logs: list[RawSQLLog]):
        self._logs = logs

    def collect(self, **kwargs) -> list[RawSQLLog]:
        """저장된 로그 반환."""
        return self._logs


class MockDescriptionGenerator:
    """설명 생성기 Mock (데모용)."""

    def __init__(self, llm_client):
        self._llm_client = llm_client

    def generate(self, sql: str) -> str:
        """SQL 설명 생성."""
        return self._llm_client.invoke(sql)


def load_sample_logs(sample_path: Path) -> list[RawSQLLog]:
    """샘플 JSON 파일에서 로그를 로드."""
    with open(sample_path, encoding="utf-8") as f:
        data = json.load(f)

    logs = []
    for item in data:
        log = RawSQLLog(
            sql_id=item["sql_id"],
            sql_text=item["sql_text"],
            exec_count=item["exec_count"],
            error_count=item["error_count"],
            collected_at=datetime.fromisoformat(item["collected_at"]),
            schema_name=item.get("schema_name"),
        )
        logs.append(log)
    return logs


def run_demo_pipeline(limit: int | None = None, dry_run: bool = False) -> PipelineResult:
    """샘플 데이터로 데모 파이프라인 실행."""
    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]🚀 Text2SQL 오프라인 파이프라인 데모 실행[/bold cyan]")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")

    # 샘플 데이터 로드
    sample_path = PROJECT_ROOT / "data" / "samples" / "sql_logs.json"
    if not sample_path.exists():
        console.print(f"[red]❌ 샘플 파일을 찾을 수 없습니다: {sample_path}[/red]")
        sys.exit(1)

    console.print(f"\n[green]📂 샘플 데이터 로드:[/green] {sample_path}")
    logs = load_sample_logs(sample_path)
    if limit:
        logs = logs[:limit]
    console.print(f"   - 로드된 로그 수: [yellow]{len(logs)}[/yellow]개")

    # Mock 컴포넌트 생성
    log_collector = MockLogCollector(logs)
    log_filter = LogFilter()
    sql_normalizer = SQLNormalizer()
    llm_client = MockLLMClient()
    description_generator = MockDescriptionGenerator(llm_client)
    vector_indexer = MockVectorIndexer(dry_run=dry_run)
    es_indexer = MockESIndexer(dry_run=dry_run)

    # 진행 상황 UI 생성
    progress_ui = PipelineProgressUI()

    # 파이프라인 생성 및 실행
    pipeline = OfflinePipeline(
        log_collector=log_collector,
        log_filter=log_filter,
        sql_normalizer=sql_normalizer,
        description_generator=description_generator,
        vector_indexer=vector_indexer,
        es_indexer=es_indexer,
        progress_callback=progress_ui.update,
    )

    console.print("\n[bold]⚙️  파이프라인 실행 중...[/bold]\n")

    # Live 디스플레이로 진행 상황 표시
    with Live(progress_ui.generate_display(), refresh_per_second=4, console=console) as live:
        def update_display(info: ProgressInfo):
            progress_ui.update(info)
            live.update(progress_ui.generate_display())

        pipeline._progress_callback = update_display
        result = pipeline.run()

    # 결과 출력
    console.print("\n")
    print_result_panel(result)

    return result


def print_result_panel(result: PipelineResult) -> None:
    """결과를 패널로 출력."""
    result_table = Table(show_header=False, box=None)
    result_table.add_column("항목", style="cyan")
    result_table.add_column("값", style="yellow")

    result_table.add_row("📥 수집된 로그", f"{result.collected_count}건")
    result_table.add_row("🔍 필터링된 로그", f"{result.filtered_count}건")
    result_table.add_row("⚙️  정규화된 템플릿", f"{result.normalized_count}건")
    result_table.add_row("📦 인덱싱된 템플릿", f"{result.indexed_count}건")
    result_table.add_row("❌ 에러 수", f"{len(result.errors)}건")

    status = "[bold green]✅ 성공[/bold green]" if result.success else "[bold red]❌ 실패[/bold red]"
    result_table.add_row("📊 결과", status)

    console.print(Panel(result_table, title="[bold blue]파이프라인 실행 결과[/bold blue]", border_style="green" if result.success else "red"))

    if result.errors:
        error_table = Table(show_header=True, header_style="bold red")
        error_table.add_column("SQL ID")
        error_table.add_column("단계")
        error_table.add_column("에러")

        for error in result.errors:
            error_table.add_row(
                error.get("sql_id", "N/A"),
                error.get("stage", "N/A"),
                error.get("error", "Unknown")
            )

        console.print(Panel(error_table, title="[bold red]에러 목록[/bold red]", border_style="red"))


def run_production_pipeline(
    settings: Settings, limit: int | None = None, dry_run: bool = False
) -> PipelineResult:
    """실제 인프라 연결 파이프라인 실행.

    주의: 실제 Oracle, Milvus, Elasticsearch, OpenAI 연결이 필요합니다.
    """
    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]🚀 Text2SQL 오프라인 파이프라인 실행[/bold cyan]")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")

    try:
        from text2sql.adapters.vector_store.milvus_adapter import MilvusAdapter
        from text2sql.adapters.search.es_adapter import ElasticsearchAdapter
        from text2sql.adapters.llm.openai_client import OpenAIClient
        from text2sql.offline.ingestor.log_collector import JsonLogCollector
        from text2sql.offline.processor.description_generator import DescriptionGenerator
        from text2sql.offline.indexer.vector_indexer import VectorIndexer
        from text2sql.offline.indexer.es_indexer import ESIndexer
        from text2sql.offline.indexer.embedding_service import EmbeddingService

        console.print("\n[bold]📡 외부 시스템 연결 중...[/bold]")

        # JSON 로그 파일 경로
        sample_path = PROJECT_ROOT / "data" / "samples" / "sql_logs.json"
        if not sample_path.exists():
            console.print(f"[red]❌ SQL 로그 파일을 찾을 수 없습니다: {sample_path}[/red]")
            sys.exit(1)

        # 연결 상태 테이블
        conn_table = Table(show_header=False, box=None)
        conn_table.add_column("서비스", width=20)
        conn_table.add_column("상태")

        # JSON 로그 파일
        conn_table.add_row("SQL 로그", f"[green]✅[/green] {sample_path}")

        # Milvus 연결
        milvus_adapter = MilvusAdapter(settings)
        conn_table.add_row("Milvus", f"[green]✅[/green] {settings.milvus_host}:{settings.milvus_port}")

        # Elasticsearch 연결
        es_adapter = ElasticsearchAdapter(settings)
        conn_table.add_row("Elasticsearch", f"[green]✅[/green] {settings.es_host}:{settings.es_port}")

        # LLM 클라이언트 (LM Studio)
        llm_client = OpenAIClient(settings)
        conn_table.add_row("LLM", f"[green]✅[/green] {settings.llm_model}")

        # 임베딩 서비스 (LM Studio)
        embedding_service = EmbeddingService(settings)
        conn_table.add_row("Embedding", f"[green]✅[/green] {settings.embedding_model}")

        console.print(Panel(conn_table, title="[bold blue]연결 상태[/bold blue]", border_style="blue"))

        # 컴포넌트 생성
        log_collector = JsonLogCollector(sample_path, limit=limit)
        log_filter = LogFilter()
        sql_normalizer = SQLNormalizer()
        description_generator = DescriptionGenerator(llm_client)
        vector_indexer = VectorIndexer(
            embedding_service=embedding_service,
            milvus_adapter=milvus_adapter,
            collection_name=settings.milvus_collection_name,
        )
        es_indexer = ESIndexer(
            es_adapter=es_adapter,
            index_name=settings.es_index_name,
        )

        # 진행 상황 UI 생성
        progress_ui = PipelineProgressUI()

        # 파이프라인 생성 및 실행
        pipeline = OfflinePipeline(
            log_collector=log_collector,
            log_filter=log_filter,
            sql_normalizer=sql_normalizer,
            description_generator=description_generator,
            vector_indexer=vector_indexer,
            es_indexer=es_indexer,
            progress_callback=progress_ui.update,
        )

        console.print("\n[bold]⚙️  파이프라인 실행 중...[/bold]\n")

        # Live 디스플레이로 진행 상황 표시
        with Live(progress_ui.generate_display(), refresh_per_second=4, console=console) as live:
            def update_display(info: ProgressInfo):
                progress_ui.update(info)
                live.update(progress_ui.generate_display())

            pipeline._progress_callback = update_display
            result = pipeline.run()

        # 결과 출력
        console.print("\n")
        print_result_panel(result)

        return result

    except ImportError as e:
        console.print(f"\n[red]❌ 필수 모듈을 임포트할 수 없습니다: {e}[/red]")
        console.print("   [dim]데모 모드(--demo)로 실행해 보세요.[/dim]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ 파이프라인 실행 실패: {e}[/red]")
        sys.exit(1)


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="Text2SQL 오프라인 파이프라인 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/run_pipeline.py --demo              # 샘플 데이터로 데모 실행
  python scripts/run_pipeline.py --demo --dry-run    # 저장 없이 테스트
  python scripts/run_pipeline.py --limit 100         # 실제 DB에서 100개만 처리
        """,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="샘플 데이터로 데모 실행 (외부 인프라 불필요)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 저장 없이 테스트 실행",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리할 최대 로그 수",
    )

    args = parser.parse_args()

    # --dry-run만 사용하면 --demo도 함께 활성화 (외부 인프라 없이 테스트)
    if args.dry_run and not args.demo:
        print("💡 --dry-run 옵션이 활성화되어 샘플 데이터로 테스트를 실행합니다.")
        args.demo = True

    if args.demo:
        result = run_demo_pipeline(limit=args.limit, dry_run=args.dry_run)
    else:
        settings = Settings()
        result = run_production_pipeline(settings, limit=args.limit, dry_run=args.dry_run)

    # 결과 코드 반환
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
