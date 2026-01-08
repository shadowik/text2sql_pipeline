"""Core 설정 모듈 테스트."""


class TestSettings:
    """Settings Pydantic 모델 테스트."""

    def test_create_settings_with_defaults(self):
        """기본값으로 Settings를 생성할 수 있어야 한다."""
        from text2sql.core.config import Settings

        settings = Settings()

        assert settings is not None
        assert hasattr(settings, "oracle_host")
        assert hasattr(settings, "oracle_port")
        assert hasattr(settings, "milvus_host")
        assert hasattr(settings, "milvus_port")
        assert hasattr(settings, "es_host")
        assert hasattr(settings, "es_port")
        assert hasattr(settings, "openai_api_key")

    def test_settings_has_oracle_config(self):
        """Settings는 Oracle 데이터베이스 설정을 가져야 한다."""
        from text2sql.core.config import Settings

        settings = Settings()

        assert settings.oracle_host is not None
        assert settings.oracle_port is not None

    def test_settings_has_vector_store_config(self):
        """Settings는 벡터 저장소(Milvus) 설정을 가져야 한다."""
        from text2sql.core.config import Settings

        settings = Settings()

        assert settings.milvus_host is not None
        assert settings.milvus_port is not None

    def test_settings_has_es_config(self):
        """Settings는 Elasticsearch 설정을 가져야 한다."""
        from text2sql.core.config import Settings

        settings = Settings()

        assert settings.es_host is not None
        assert settings.es_port is not None


class TestSettingsFromEnv:
    """환경변수에서 설정 로드 테스트."""

    def test_load_oracle_host_from_env(self, monkeypatch):
        """환경변수에서 Oracle 호스트를 로드할 수 있어야 한다."""
        monkeypatch.setenv("TEXT2SQL_ORACLE_HOST", "oracle.example.com")

        from text2sql.core.config import Settings

        settings = Settings()

        assert settings.oracle_host == "oracle.example.com"

    def test_load_milvus_config_from_env(self, monkeypatch):
        """환경변수에서 Milvus 설정을 로드할 수 있어야 한다."""
        monkeypatch.setenv("TEXT2SQL_MILVUS_HOST", "milvus.example.com")
        monkeypatch.setenv("TEXT2SQL_MILVUS_PORT", "19531")

        from text2sql.core.config import Settings

        settings = Settings()

        assert settings.milvus_host == "milvus.example.com"
        assert settings.milvus_port == 19531

    def test_load_openai_api_key_from_env(self, monkeypatch):
        """환경변수에서 OpenAI API 키를 로드할 수 있어야 한다."""
        monkeypatch.setenv("TEXT2SQL_OPENAI_API_KEY", "sk-test-key-123")

        from text2sql.core.config import Settings

        settings = Settings()

        assert settings.openai_api_key == "sk-test-key-123"


class TestSettingsDefaults:
    """기본값 적용 테스트."""

    def test_oracle_defaults(self):
        """Oracle 설정의 기본값이 올바르게 적용되어야 한다."""
        from text2sql.core.config import Settings

        settings = Settings()

        assert settings.oracle_host == "localhost"
        assert settings.oracle_port == 1521
        assert settings.oracle_service_name == "ORCL"

    def test_milvus_defaults(self):
        """Milvus 설정의 기본값이 올바르게 적용되어야 한다."""
        from text2sql.core.config import Settings

        settings = Settings()

        assert settings.milvus_host == "localhost"
        assert settings.milvus_port == 19530
        assert settings.milvus_collection_name == "sql_templates"

    def test_es_defaults(self):
        """Elasticsearch 설정의 기본값이 올바르게 적용되어야 한다."""
        from text2sql.core.config import Settings

        settings = Settings()

        assert settings.es_host == "localhost"
        assert settings.es_port == 9200
        assert settings.es_index_name == "sql_templates"

    def test_openai_defaults(self):
        """OpenAI 설정의 기본값이 올바르게 적용되어야 한다."""
        from text2sql.core.config import Settings

        settings = Settings()

        assert settings.openai_api_key is None
        assert settings.openai_model == "gpt-4o"

