#!/usr/bin/env python
"""테이블 설명 생성 LLM 프롬프트 테스트 스크립트.

이 스크립트는 불충분한 테이블/컬럼 코멘트를 포함한 샘플 데이터를 로드하고,
LLM 프롬프트를 생성하여 출력합니다.

실제 LLM 호출은 사용자가 직접 수행합니다.

사용법:
    # 모든 테이블에 대한 프롬프트 생성
    python experiments/test_table_description_prompt.py

    # 특정 테이블에 대한 프롬프트만 생성
    python experiments/test_table_description_prompt.py --table MES_EQUIP_STAT

    # 컬럼 설명 프롬프트 생성
    python experiments/test_table_description_prompt.py --table MES_EQUIP_STAT --column STAT_CD

    # 파일로 출력
    python experiments/test_table_description_prompt.py --output prompts_output.txt
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from text2sql.offline.processor.table_description_prompt_builder import (
    ColumnRegistry,
    JoinPattern,
    TableDescriptionPromptBuilder,
    TableRegistry,
    WherePattern,
    load_sample_data,
)


def print_separator(title: str, char: str = "=", width: int = 80) -> str:
    """구분선 출력."""
    line = char * width
    return f"\n{line}\n{title}\n{line}\n"


def analyze_comment_quality(
    builder: TableDescriptionPromptBuilder,
    table_registries: list[TableRegistry],
    column_registries_by_table: dict[str, list[ColumnRegistry]],
) -> str:
    """코멘트 품질 분석 결과를 출력합니다."""
    output = []
    output.append(print_separator("코멘트 품질 분석 결과"))

    for table in table_registries:
        is_insufficient = builder.is_comment_insufficient(table.system_comment)
        status = "❌ 불충분" if is_insufficient else "✅ 충분"
        comment_display = table.system_comment if table.system_comment else "(없음)"
        output.append(f"\n[테이블] {table.table_name}")
        output.append(f"  - system_comment: {comment_display}")
        output.append(f"  - 상태: {status}")

        if table.table_name in column_registries_by_table:
            insufficient_cols = []
            for col in column_registries_by_table[table.table_name]:
                if builder.is_comment_insufficient(col.system_comment):
                    insufficient_cols.append(col.column_name)

            if insufficient_cols:
                output.append(f"  - 코멘트 보강 필요 컬럼: {', '.join(insufficient_cols)}")

    return "\n".join(output)


def generate_table_prompt(
    builder: TableDescriptionPromptBuilder,
    table_registry: TableRegistry,
    column_registries: list[ColumnRegistry],
    sql_patterns: dict,
    sample_data: list[dict],
) -> str:
    """특정 테이블에 대한 프롬프트를 생성합니다."""
    # JOIN 패턴 파싱
    join_patterns = []
    for jp in sql_patterns.get("join_patterns", []):
        join_patterns.append(
            JoinPattern(
                target_table=jp["target_table"],
                join_column=jp["join_column"],
                join_count=jp["join_count"],
            )
        )

    # WHERE 패턴 파싱
    where_patterns = []
    for wp in sql_patterns.get("where_patterns", []):
        where_patterns.append(
            WherePattern(condition=wp["condition"], count=wp["count"])
        )

    # 프롬프트 생성
    prompt = builder.build_table_description_prompt(
        table_registry=table_registry,
        column_registries=column_registries,
        sample_sqls=sql_patterns.get("sample_sqls", []),
        join_patterns=join_patterns,
        where_patterns=where_patterns,
        sample_data=sample_data,
    )

    return prompt


def generate_column_prompt(
    builder: TableDescriptionPromptBuilder,
    table_name: str,
    column_registry: ColumnRegistry,
    sample_sqls: list[str],
) -> str:
    """특정 컬럼에 대한 프롬프트를 생성합니다."""
    # 해당 컬럼이 사용된 SQL만 필터링
    column_sqls = [
        sql for sql in sample_sqls if column_registry.column_name in sql.upper()
    ]

    return builder.build_column_description_prompt(
        table_name=table_name,
        column_registry=column_registry,
        sample_sqls=column_sqls[:5],
    )


def main():
    parser = argparse.ArgumentParser(
        description="테이블 설명 생성 LLM 프롬프트 테스트"
    )
    parser.add_argument(
        "--table",
        type=str,
        help="특정 테이블에 대한 프롬프트만 생성 (예: MES_EQUIP_STAT)",
    )
    parser.add_argument(
        "--column",
        type=str,
        help="특정 컬럼에 대한 프롬프트 생성 (--table과 함께 사용)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="출력 파일 경로 (지정하지 않으면 콘솔 출력)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ko",
        choices=["ko", "en"],
        help="출력 언어 (ko: 한국어, en: 영어)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="코멘트 품질 분석만 수행",
    )

    args = parser.parse_args()

    # 샘플 데이터 경로
    sample_data_path = os.path.join(
        os.path.dirname(__file__), "sample_data"
    )

    # 샘플 데이터 로드
    print("샘플 데이터 로드 중...")
    (
        table_registries,
        column_registries_by_table,
        sql_patterns_by_table,
        sample_data_by_table,
    ) = load_sample_data(sample_data_path)

    print(f"  - 로드된 테이블 수: {len(table_registries)}")
    print(f"  - 테이블 목록: {[t.table_name for t in table_registries]}")

    # 프롬프트 빌더 생성
    builder = TableDescriptionPromptBuilder(language=args.language)

    output_lines = []

    # 코멘트 품질 분석
    analysis_result = analyze_comment_quality(
        builder, table_registries, column_registries_by_table
    )
    output_lines.append(analysis_result)

    if args.analyze_only:
        result = "\n".join(output_lines)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"\n결과가 {args.output}에 저장되었습니다.")
        else:
            print(result)
        return

    # 프롬프트 생성
    if args.column and args.table:
        # 특정 컬럼에 대한 프롬프트
        output_lines.append(
            print_separator(f"컬럼 설명 프롬프트: {args.table}.{args.column}")
        )

        if args.table not in column_registries_by_table:
            print(f"오류: 테이블 '{args.table}'을 찾을 수 없습니다.")
            return

        column_registry = None
        for col in column_registries_by_table[args.table]:
            if col.column_name == args.column:
                column_registry = col
                break

        if column_registry is None:
            print(f"오류: 컬럼 '{args.column}'을 찾을 수 없습니다.")
            return

        sample_sqls = sql_patterns_by_table.get(args.table, {}).get("sample_sqls", [])
        prompt = generate_column_prompt(
            builder, args.table, column_registry, sample_sqls
        )
        output_lines.append(prompt)

    elif args.table:
        # 특정 테이블에 대한 프롬프트
        output_lines.append(print_separator(f"테이블 설명 프롬프트: {args.table}"))

        table_registry = None
        for t in table_registries:
            if t.table_name == args.table:
                table_registry = t
                break

        if table_registry is None:
            print(f"오류: 테이블 '{args.table}'을 찾을 수 없습니다.")
            return

        column_registries = column_registries_by_table.get(args.table, [])
        sql_patterns = sql_patterns_by_table.get(args.table, {})
        sample_data = sample_data_by_table.get(args.table, [])

        prompt = generate_table_prompt(
            builder, table_registry, column_registries, sql_patterns, sample_data
        )
        output_lines.append(prompt)

    else:
        # 모든 테이블에 대한 프롬프트
        for table_registry in table_registries:
            table_name = table_registry.table_name

            # 코멘트가 불충분한 테이블만 처리
            if not builder.is_comment_insufficient(table_registry.system_comment):
                output_lines.append(
                    f"\n[SKIP] {table_name}: 코멘트가 충분함 - '{table_registry.system_comment}'"
                )
                continue

            output_lines.append(print_separator(f"테이블 설명 프롬프트: {table_name}"))

            column_registries = column_registries_by_table.get(table_name, [])
            sql_patterns = sql_patterns_by_table.get(table_name, {})
            sample_data = sample_data_by_table.get(table_name, [])

            prompt = generate_table_prompt(
                builder, table_registry, column_registries, sql_patterns, sample_data
            )
            output_lines.append(prompt)

    # 결과 출력
    result = "\n".join(output_lines)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\n결과가 {args.output}에 저장되었습니다.")
        print(f"파일 크기: {len(result):,} bytes")
    else:
        print(result)


if __name__ == "__main__":
    main()
