#!/usr/bin/env python
"""데이터 초기화 스크립트.

Milvus 컬렉션과 Elasticsearch 인덱스를 삭제합니다.

사용법:
    python scripts/reset_data.py
    python scripts/reset_data.py --milvus-only   # Milvus만 초기화
    python scripts/reset_data.py --es-only       # ES만 초기화
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

console = Console()


def reset_milvus(collection_name: str = "sql_templates") -> bool:
    """Milvus 컬렉션을 삭제합니다."""
    try:
        from pymilvus import utility, connections
        
        console.print("[yellow]Milvus 연결 중...[/yellow]")
        connections.connect(host="localhost", port="19530")
        
        if utility.has_collection(collection_name):
            console.print(f"[yellow]컬렉션 '{collection_name}' 삭제 중...[/yellow]")
            utility.drop_collection(collection_name)
            console.print(f"[green]✅ Milvus 컬렉션 '{collection_name}' 삭제 완료[/green]")
        else:
            console.print(f"[dim]컬렉션 '{collection_name}'이(가) 존재하지 않습니다.[/dim]")
        
        connections.disconnect("default")
        return True
    except Exception as e:
        console.print(f"[red]❌ Milvus 초기화 실패: {e}[/red]")
        return False


def reset_elasticsearch(index_name: str = "sql_templates") -> bool:
    """Elasticsearch 인덱스를 삭제합니다."""
    try:
        from elasticsearch import Elasticsearch
        
        console.print("[yellow]Elasticsearch 연결 중...[/yellow]")
        es = Elasticsearch(["http://localhost:9200"])
        
        # sql_templates 인덱스 삭제
        if es.indices.exists(index=index_name):
            console.print(f"[yellow]인덱스 '{index_name}' 삭제 중...[/yellow]")
            es.indices.delete(index=index_name)
            console.print(f"[green]✅ ES 인덱스 '{index_name}' 삭제 완료[/green]")
        else:
            console.print(f"[dim]인덱스 '{index_name}'이(가) 존재하지 않습니다.[/dim]")
        
        # text2sql_ 프리픽스 인덱스들도 삭제
        indices = es.indices.get_alias(index="text2sql_*")
        for idx in indices.keys():
            console.print(f"[yellow]인덱스 '{idx}' 삭제 중...[/yellow]")
            es.indices.delete(index=idx)
            console.print(f"[green]✅ ES 인덱스 '{idx}' 삭제 완료[/green]")
        
        return True
    except Exception as e:
        console.print(f"[red]❌ Elasticsearch 초기화 실패: {e}[/red]")
        return False


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="Milvus 및 Elasticsearch 데이터 초기화"
    )
    parser.add_argument(
        "--milvus-only",
        action="store_true",
        help="Milvus만 초기화",
    )
    parser.add_argument(
        "--es-only",
        action="store_true",
        help="Elasticsearch만 초기화",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="sql_templates",
        help="Milvus 컬렉션 이름 (기본값: sql_templates)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="sql_templates",
        help="ES 인덱스 이름 (기본값: sql_templates)",
    )

    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]🗑️  데이터 초기화 스크립트[/bold cyan]\n"
        "Milvus 컬렉션과 Elasticsearch 인덱스를 삭제합니다.",
        border_style="blue"
    ))

    results = []

    if not args.es_only:
        milvus_result = reset_milvus(args.collection)
        results.append(("Milvus", milvus_result))

    if not args.milvus_only:
        es_result = reset_elasticsearch(args.index)
        results.append(("Elasticsearch", es_result))

    # 결과 요약
    console.print("\n")
    result_table = Table(title="초기화 결과", show_header=True)
    result_table.add_column("서비스", style="cyan")
    result_table.add_column("결과")

    all_success = True
    for service, success in results:
        status = "[green]✅ 성공[/green]" if success else "[red]❌ 실패[/red]"
        result_table.add_row(service, status)
        if not success:
            all_success = False

    console.print(result_table)

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
