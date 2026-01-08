"""SQL 정규화기 테스트."""


class TestNormalizeLiterals:
    """리터럴 치환 테스트."""

    def test_replace_string_literal_with_placeholder(self):
        """문자열 리터럴을 placeholder로 치환해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM users WHERE name = 'John'"

        result = normalizer.normalize_literals(sql)

        assert "'John'" not in result
        assert ":placeholder" in result or "?" in result

    def test_replace_multiple_string_literals(self):
        """여러 문자열 리터럴을 모두 치환해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM users WHERE name = 'John' AND city = 'Seoul'"

        result = normalizer.normalize_literals(sql)

        assert "'John'" not in result
        assert "'Seoul'" not in result

    def test_handle_escaped_quotes_in_string(self):
        """이스케이프된 따옴표가 있는 문자열도 처리해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM users WHERE name = 'O''Brien'"

        result = normalizer.normalize_literals(sql)

        assert "'O''Brien'" not in result

    def test_replace_number_literal_with_placeholder(self):
        """숫자 리터럴을 placeholder로 치환해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM users WHERE id = 123"

        result = normalizer.normalize_literals(sql)

        assert "123" not in result
        assert ":placeholder" in result

    def test_replace_decimal_number_literal(self):
        """소수점 숫자 리터럴을 치환해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM products WHERE price > 99.99"

        result = normalizer.normalize_literals(sql)

        assert "99.99" not in result

    def test_preserve_column_names_with_numbers(self):
        """숫자가 포함된 컬럼명은 보존해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT col1, col2 FROM table1 WHERE id = 5"

        result = normalizer.normalize_literals(sql)

        assert "col1" in result
        assert "col2" in result
        assert "table1" in result
        assert " 5" not in result  # 숫자 5는 치환되어야 함

    def test_replace_date_literal_with_placeholder(self):
        """DATE 함수 내의 날짜 리터럴을 치환해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM orders WHERE order_date = DATE'2024-01-15'"

        result = normalizer.normalize_literals(sql)

        assert "2024-01-15" not in result

    def test_replace_to_date_function_literal(self):
        """TO_DATE 함수 내의 날짜 리터럴을 치환해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM orders WHERE order_date = TO_DATE('2024-01-15', 'YYYY-MM-DD')"

        result = normalizer.normalize_literals(sql)

        assert "'2024-01-15'" not in result
        # 포맷 문자열도 문자열이므로 치환됨
        assert "'YYYY-MM-DD'" not in result

    def test_replace_in_clause_values_with_placeholder(self):
        """IN 절의 여러 값들을 placeholder로 치환해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM users WHERE id IN (1, 2, 3)"

        result = normalizer.normalize_literals(sql)

        assert "1" not in result.replace("placeholder", "")
        assert "2" not in result.replace("placeholder", "")
        assert "3" not in result.replace("placeholder", "")

    def test_replace_in_clause_string_values(self):
        """IN 절의 문자열 값들을 placeholder로 치환해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM users WHERE status IN ('active', 'pending', 'inactive')"

        result = normalizer.normalize_literals(sql)

        assert "'active'" not in result
        assert "'pending'" not in result
        assert "'inactive'" not in result

    def test_preserve_existing_bind_variables(self):
        """이미 바인드 변수(:1, :2)가 있는 SQL은 그대로 유지해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM users WHERE id = :1 AND name = :2"

        result = normalizer.normalize_literals(sql)

        assert ":1" in result
        assert ":2" in result

    def test_preserve_named_bind_variables(self):
        """이름 있는 바인드 변수(:id, :name)가 있는 SQL은 그대로 유지해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM users WHERE id = :id AND name = :name"

        result = normalizer.normalize_literals(sql)

        assert ":id" in result
        assert ":name" in result


class TestExtractTables:
    """테이블명 추출 테스트."""

    def test_extract_single_table_from_simple_select(self):
        """단순 SELECT에서 단일 테이블을 추출해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM users WHERE id = 1"

        tables = normalizer.extract_tables(sql)

        assert "users" in tables

    def test_extract_table_with_schema_prefix(self):
        """스키마 접두사가 있는 테이블을 추출해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM schema1.users WHERE id = 1"

        tables = normalizer.extract_tables(sql)

        assert "schema1.users" in tables or "users" in tables

    def test_extract_table_with_alias(self):
        """별칭이 있는 테이블을 추출해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT u.id FROM users u WHERE u.id = 1"

        tables = normalizer.extract_tables(sql)

        assert "users" in tables

    def test_extract_tables_from_inner_join(self):
        """INNER JOIN에서 여러 테이블을 추출해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id"

        tables = normalizer.extract_tables(sql)

        assert "users" in tables
        assert "orders" in tables

    def test_extract_tables_from_left_join(self):
        """LEFT JOIN에서 여러 테이블을 추출해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM users u LEFT JOIN orders o ON u.id = o.user_id"

        tables = normalizer.extract_tables(sql)

        assert "users" in tables
        assert "orders" in tables

    def test_extract_tables_from_multiple_joins(self):
        """여러 JOIN이 있는 경우 모든 테이블을 추출해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = """
            SELECT * FROM users u
            INNER JOIN orders o ON u.id = o.user_id
            LEFT JOIN products p ON o.product_id = p.id
        """

        tables = normalizer.extract_tables(sql)

        assert "users" in tables
        assert "orders" in tables
        assert "products" in tables

    def test_extract_tables_from_subquery(self):
        """서브쿼리에서도 테이블을 추출해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = """
            SELECT * FROM users u
            WHERE u.id IN (SELECT user_id FROM orders WHERE status = 'active')
        """

        tables = normalizer.extract_tables(sql)

        assert "users" in tables
        assert "orders" in tables

    def test_extract_tables_from_nested_subquery(self):
        """중첩 서브쿼리에서도 테이블을 추출해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = """
            SELECT * FROM users u
            WHERE u.id IN (
                SELECT user_id FROM orders
                WHERE product_id IN (SELECT id FROM products WHERE category = 'A')
            )
        """

        tables = normalizer.extract_tables(sql)

        assert "users" in tables
        assert "orders" in tables
        assert "products" in tables


class TestExtractColumns:
    """컬럼명 추출 테스트."""

    def test_extract_columns_from_select(self):
        """SELECT 절에서 컬럼명을 추출해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT id, name, email FROM users"

        columns = normalizer.extract_columns(sql)

        assert "id" in columns
        assert "name" in columns
        assert "email" in columns

    def test_extract_columns_with_table_prefix(self):
        """테이블 접두사가 있는 컬럼명을 추출해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT u.id, u.name FROM users u"

        columns = normalizer.extract_columns(sql)

        assert "id" in columns
        assert "name" in columns

    def test_extract_star_as_column(self):
        """SELECT *도 컬럼으로 인식해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT * FROM users"

        columns = normalizer.extract_columns(sql)

        assert "*" in columns

    def test_extract_columns_from_where_clause(self):
        """WHERE 절에서 사용된 컬럼을 추출해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = "SELECT id FROM users WHERE status = 'active' AND created_at > :date"

        columns = normalizer.extract_where_columns(sql)

        assert "status" in columns
        assert "created_at" in columns


class TestUnionAll:
    """UNION ALL 패턴 테스트."""

    def test_extract_tables_from_union_all(self):
        """UNION ALL 쿼리에서 모든 테이블을 추출해야 한다."""
        from text2sql.offline.processor.sql_normalizer import SQLNormalizer

        normalizer = SQLNormalizer()
        sql = """
            SELECT id, name FROM users
            UNION ALL
            SELECT id, name FROM admins
            UNION ALL
            SELECT id, name FROM guests
        """

        tables = normalizer.extract_tables(sql)

        assert "users" in tables
        assert "admins" in tables
        assert "guests" in tables
