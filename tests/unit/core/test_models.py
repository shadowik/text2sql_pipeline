"""Core 모듈 데이터 모델 테스트."""

from datetime import datetime


class TestRawSQLLog:
    """RawSQLLog 데이터클래스 테스트."""

    def test_create_raw_sql_log_with_required_fields(self):
        """필수 필드로 RawSQLLog를 생성할 수 있어야 한다."""
        from text2sql.core.models import RawSQLLog

        log = RawSQLLog(
            sql_id="sql_001",
            sql_text="SELECT * FROM users WHERE id = :1",
            exec_count=100,
            error_count=0,
            collected_at=datetime(2026, 1, 8, 10, 0, 0),
        )

        assert log.sql_id == "sql_001"
        assert log.sql_text == "SELECT * FROM users WHERE id = :1"
        assert log.exec_count == 100
        assert log.error_count == 0
        assert log.collected_at == datetime(2026, 1, 8, 10, 0, 0)

    def test_raw_sql_log_has_optional_schema_name(self):
        """RawSQLLog는 선택적으로 schema_name을 가질 수 있어야 한다."""
        from text2sql.core.models import RawSQLLog

        log = RawSQLLog(
            sql_id="sql_002",
            sql_text="SELECT * FROM orders",
            exec_count=50,
            error_count=0,
            collected_at=datetime(2026, 1, 8, 10, 0, 0),
            schema_name="PROD_SCHEMA",
        )

        assert log.schema_name == "PROD_SCHEMA"

    def test_raw_sql_log_optional_fields_default_to_none(self):
        """RawSQLLog의 선택적 필드는 기본값으로 None이어야 한다."""
        from text2sql.core.models import RawSQLLog

        log = RawSQLLog(
            sql_id="sql_003",
            sql_text="SELECT 1 FROM dual",
            exec_count=10,
            error_count=0,
            collected_at=datetime(2026, 1, 8, 10, 0, 0),
        )

        assert log.schema_name is None


class TestNormalizedSQL:
    """NormalizedSQL 데이터클래스 테스트."""

    def test_create_normalized_sql_with_required_fields(self):
        """필수 필드로 NormalizedSQL을 생성할 수 있어야 한다."""
        from text2sql.core.models import NormalizedSQL

        normalized = NormalizedSQL(
            original_sql_id="sql_001",
            normalized_text="SELECT * FROM users WHERE id = :placeholder",
            tables=["users"],
            columns=["id"],
        )

        assert normalized.original_sql_id == "sql_001"
        assert normalized.normalized_text == "SELECT * FROM users WHERE id = :placeholder"
        assert normalized.tables == ["users"]
        assert normalized.columns == ["id"]

    def test_normalized_sql_has_template_hash(self):
        """NormalizedSQL은 template_hash를 가질 수 있어야 한다."""
        from text2sql.core.models import NormalizedSQL

        normalized = NormalizedSQL(
            original_sql_id="sql_002",
            normalized_text="SELECT name FROM products WHERE price > :placeholder",
            tables=["products"],
            columns=["name", "price"],
            template_hash="abc123",
        )

        assert normalized.template_hash == "abc123"

    def test_normalized_sql_optional_fields_default_to_none(self):
        """NormalizedSQL의 선택적 필드는 기본값으로 None이어야 한다."""
        from text2sql.core.models import NormalizedSQL

        normalized = NormalizedSQL(
            original_sql_id="sql_003",
            normalized_text="SELECT 1 FROM dual",
            tables=["dual"],
            columns=[],
        )

        assert normalized.template_hash is None


class TestSQLTemplate:
    """SQLTemplate 데이터클래스 테스트."""

    def test_create_sql_template_with_required_fields(self):
        """필수 필드로 SQLTemplate을 생성할 수 있어야 한다."""
        from text2sql.core.models import SQLTemplate

        template = SQLTemplate(
            template_id="tpl_001",
            template_text="SELECT * FROM users WHERE id = :placeholder",
            description="사용자 ID로 사용자 정보를 조회한다",
            tables=["users"],
            columns=["id"],
        )

        assert template.template_id == "tpl_001"
        assert template.template_text == "SELECT * FROM users WHERE id = :placeholder"
        assert template.description == "사용자 ID로 사용자 정보를 조회한다"
        assert template.tables == ["users"]
        assert template.columns == ["id"]

    def test_sql_template_has_embedding_vector(self):
        """SQLTemplate은 embedding 벡터를 가질 수 있어야 한다."""
        from text2sql.core.models import SQLTemplate

        embedding = [0.1, 0.2, 0.3]
        template = SQLTemplate(
            template_id="tpl_002",
            template_text="SELECT name FROM products",
            description="상품명 조회",
            tables=["products"],
            columns=["name"],
            embedding=embedding,
        )

        assert template.embedding == embedding

    def test_sql_template_has_exec_count(self):
        """SQLTemplate은 실행 횟수를 가질 수 있어야 한다."""
        from text2sql.core.models import SQLTemplate

        template = SQLTemplate(
            template_id="tpl_003",
            template_text="SELECT 1 FROM dual",
            description="테스트 쿼리",
            tables=["dual"],
            columns=[],
            exec_count=1000,
        )

        assert template.exec_count == 1000

    def test_sql_template_optional_fields_have_defaults(self):
        """SQLTemplate의 선택적 필드는 기본값을 가져야 한다."""
        from text2sql.core.models import SQLTemplate

        template = SQLTemplate(
            template_id="tpl_004",
            template_text="SELECT 1 FROM dual",
            description="테스트",
            tables=["dual"],
            columns=[],
        )

        assert template.embedding is None
        assert template.exec_count == 0


class TestSQLOperation:
    """SQLOperation Enum 테스트."""

    def test_sql_operation_has_select(self):
        """SQLOperation은 SELECT 값을 가져야 한다."""
        from text2sql.core.models import SQLOperation

        assert SQLOperation.SELECT.value == "SELECT"

    def test_sql_operation_has_insert(self):
        """SQLOperation은 INSERT 값을 가져야 한다."""
        from text2sql.core.models import SQLOperation

        assert SQLOperation.INSERT.value == "INSERT"

    def test_sql_operation_has_update(self):
        """SQLOperation은 UPDATE 값을 가져야 한다."""
        from text2sql.core.models import SQLOperation

        assert SQLOperation.UPDATE.value == "UPDATE"

    def test_sql_operation_has_delete(self):
        """SQLOperation은 DELETE 값을 가져야 한다."""
        from text2sql.core.models import SQLOperation

        assert SQLOperation.DELETE.value == "DELETE"

    def test_sql_operation_has_ddl_types(self):
        """SQLOperation은 DDL 타입들을 가져야 한다."""
        from text2sql.core.models import SQLOperation

        assert SQLOperation.CREATE.value == "CREATE"
        assert SQLOperation.ALTER.value == "ALTER"
        assert SQLOperation.DROP.value == "DROP"

