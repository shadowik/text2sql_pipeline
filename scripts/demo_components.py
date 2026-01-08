#!/usr/bin/env python
"""개별 컴포넌트 데모 스크립트.

파이프라인의 각 컴포넌트를 개별적으로 테스트합니다.

사용법:
    python scripts/demo_components.py normalizer    # SQL 정규화 데모
    python scripts/demo_components.py filter        # 로그 필터 데모
    python scripts/demo_components.py glossary      # 용어 사전 데모
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text2sql.core.models import RawSQLLog, GlossaryTerm
from text2sql.offline.ingestor.log_filter import LogFilter
from text2sql.offline.processor.sql_normalizer import SQLNormalizer
from text2sql.offline.schema.glossary_builder import GlossaryBuilder


def demo_normalizer():
    """SQL 정규화기 데모."""
    print("\n" + "=" * 60)
    print("🔧 SQL 정규화기 데모")
    print("=" * 60)

    normalizer = SQLNormalizer()

    sample_sqls = [
        "SELECT * FROM customers WHERE customer_id = 12345",
        "SELECT name, email FROM users WHERE created_at > '2025-01-01'",
        "SELECT * FROM products WHERE price > 100.50 AND category = 'ELECTRONICS'",
        "SELECT o.*, c.name FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.id IN (1, 2, 3, 4, 5)",
    ]

    for sql in sample_sqls:
        print(f"\n📝 원본 SQL:")
        print(f"   {sql}")

        normalized = normalizer.normalize_literals(sql)
        tables = normalizer.extract_tables(sql)
        columns = normalizer.extract_columns(sql)

        print(f"\n🔄 정규화된 SQL:")
        print(f"   {normalized}")
        print(f"\n📊 추출된 메타데이터:")
        print(f"   - 테이블: {tables}")
        print(f"   - 컬럼: {columns}")
        print("-" * 60)


def demo_filter():
    """로그 필터 데모."""
    print("\n" + "=" * 60)
    print("🔍 로그 필터 데모")
    print("=" * 60)

    # 샘플 로그 로드
    sample_path = PROJECT_ROOT / "data" / "samples" / "sql_logs.json"
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

    print(f"\n📂 로드된 전체 로그: {len(logs)}개")

    # 필터링
    log_filter = LogFilter()
    filtered = log_filter.filter(logs)

    print(f"✅ 필터링 후 로그: {len(filtered)}개")
    print(f"❌ 제외된 로그: {len(logs) - len(filtered)}개")

    # 제외된 로그 분석
    excluded = [log for log in logs if log not in filtered]
    print("\n📋 제외된 로그 상세:")
    for log in excluded:
        reason = []
        if log.error_count > 0:
            reason.append("에러 있음")
        if log.sql_text.strip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            reason.append("DML")
        if log.sql_text.strip().upper().startswith(("CREATE", "ALTER", "DROP")):
            reason.append("DDL")
        if "DBA_" in log.sql_text or "SYS." in log.sql_text:
            reason.append("시스템 쿼리")
        print(f"   - {log.sql_id}: {', '.join(reason)}")

    # 상위 5개 출력
    print("\n🏆 상위 5개 필터링된 로그:")
    for log in filtered[:5]:
        print(f"   - {log.sql_id} (exec_count: {log.exec_count})")
        print(f"     {log.sql_text[:80]}...")


def demo_glossary():
    """용어 사전 데모."""
    print("\n" + "=" * 60)
    print("📖 용어 사전 데모")
    print("=" * 60)

    # CSV 로드
    glossary_path = PROJECT_ROOT / "data" / "samples" / "glossary.csv"
    builder = GlossaryBuilder()

    print(f"\n📂 용어 사전 로드: {glossary_path}")
    raw_terms = builder.parse_csv(glossary_path)
    print(f"   - 로드된 용어 수: {len(raw_terms)}개")

    # GlossaryTerm 객체 생성
    terms = [builder.create_term(t) for t in raw_terms]

    # 카테고리별 그룹화
    categories: dict[str, list[GlossaryTerm]] = {}
    for term in terms:
        cat = term.category or "기타"
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(term)

    print("\n📊 카테고리별 용어:")
    for cat, cat_terms in sorted(categories.items()):
        print(f"\n   [{cat}] ({len(cat_terms)}개)")
        for term in cat_terms[:3]:  # 각 카테고리별 최대 3개만 출력
            print(f"     - {term.term} ({term.korean_name})")
        if len(cat_terms) > 3:
            print(f"     ... 외 {len(cat_terms) - 3}개")

    # 검색 예시
    print("\n🔍 검색 예시:")
    search_terms = ["customer", "order", "salary"]
    for search in search_terms:
        matches = [t for t in terms if search in t.term.lower()]
        if matches:
            print(f"   '{search}' 검색 결과:")
            for match in matches[:3]:
                print(f"     - {match.term}: {match.korean_name} - {match.description[:30]}...")


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="Text2SQL 개별 컴포넌트 데모",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "component",
        choices=["normalizer", "filter", "glossary", "all"],
        help="데모할 컴포넌트 선택",
    )

    args = parser.parse_args()

    if args.component == "normalizer" or args.component == "all":
        demo_normalizer()
    if args.component == "filter" or args.component == "all":
        demo_filter()
    if args.component == "glossary" or args.component == "all":
        demo_glossary()

    print("\n✅ 데모 완료!")


if __name__ == "__main__":
    main()
