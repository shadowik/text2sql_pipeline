"""SQL 정규화기 - 리터럴 치환 및 메타데이터 추출."""

import re


class SQLNormalizer:
    """SQL 정규화기."""

    # 문자열 리터럴 패턴 (이스케이프된 따옴표 포함)
    STRING_LITERAL_PATTERN = re.compile(r"'(?:[^']|'')*'")

    # 숫자 리터럴 패턴 (컬럼명/테이블명 및 바인드 변수와 구분)
    # 바인드 변수(:1, :2)는 제외
    NUMBER_LITERAL_PATTERN = re.compile(r"(?<![a-zA-Z_:])(\d+\.?\d*)(?![a-zA-Z_\d])")

    def normalize_literals(self, sql: str) -> str:
        """SQL의 리터럴을 placeholder로 치환한다.

        Args:
            sql: 원본 SQL 문자열

        Returns:
            리터럴이 placeholder로 치환된 SQL
        """
        # 먼저 문자열 리터럴 치환
        result = self.STRING_LITERAL_PATTERN.sub(":placeholder", sql)
        # 숫자 리터럴 치환
        result = self.NUMBER_LITERAL_PATTERN.sub(":placeholder", result)
        return result

    def extract_tables(self, sql: str) -> list[str]:
        """SQL에서 테이블명을 추출한다.

        Args:
            sql: SQL 문자열

        Returns:
            추출된 테이블명 리스트
        """
        tables = []

        # FROM 절 테이블 추출 패턴 (스키마.테이블 또는 테이블, 별칭 포함)
        from_pattern = re.compile(
            r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)",
            re.IGNORECASE,
        )

        # JOIN 절 테이블 추출 패턴
        join_pattern = re.compile(
            r"\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)",
            re.IGNORECASE,
        )

        for match in from_pattern.finditer(sql):
            tables.append(match.group(1))

        for match in join_pattern.finditer(sql):
            tables.append(match.group(1))

        return tables

    def extract_columns(self, sql: str) -> list[str]:
        """SQL에서 컬럼명을 추출한다.

        Args:
            sql: SQL 문자열

        Returns:
            추출된 컬럼명 리스트
        """
        columns = []

        # SELECT와 FROM 사이의 컬럼 부분 추출
        select_pattern = re.compile(
            r"\bSELECT\s+(.*?)\s+FROM\b",
            re.IGNORECASE | re.DOTALL,
        )

        match = select_pattern.search(sql)
        if match:
            columns_str = match.group(1)
            # 컬럼 분리 (쉼표로)
            for col in columns_str.split(","):
                col = col.strip()
                if col == "*":
                    columns.append("*")
                elif "." in col:
                    # 테이블.컬럼 형식에서 컬럼명만 추출
                    columns.append(col.split(".")[-1].split()[0])
                else:
                    # 별칭이 있을 수 있으므로 첫 번째 단어만
                    columns.append(col.split()[0])

        return columns

    def extract_where_columns(self, sql: str) -> list[str]:
        """SQL의 WHERE 절에서 컬럼명을 추출한다.

        Args:
            sql: SQL 문자열

        Returns:
            WHERE 절에서 추출된 컬럼명 리스트
        """
        columns = []

        # WHERE 절 추출
        where_pattern = re.compile(
            r"\bWHERE\s+(.*?)(?:\bORDER\b|\bGROUP\b|\bHAVING\b|\bLIMIT\b|$)",
            re.IGNORECASE | re.DOTALL,
        )

        match = where_pattern.search(sql)
        if match:
            where_clause = match.group(1)
            # 컬럼명 패턴: 비교 연산자 앞의 식별자
            column_pattern = re.compile(
                r"(?:^|AND|OR)\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s*(?:=|!=|<>|<|>|<=|>=|IN|LIKE|BETWEEN|IS)",
                re.IGNORECASE,
            )
            for col_match in column_pattern.finditer(where_clause):
                col = col_match.group(1)
                # 테이블.컬럼 형식에서 컬럼명만 추출
                if "." in col:
                    columns.append(col.split(".")[-1])
                else:
                    columns.append(col)

        return columns

