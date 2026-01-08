"""애플리케이션 설정 모듈."""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정."""

    # Oracle 데이터베이스 설정
    oracle_host: str = "localhost"
    oracle_port: int = 1521
    oracle_service_name: str = "ORCL"
    oracle_user: Optional[str] = None
    oracle_password: Optional[str] = None

    # Milvus 벡터 저장소 설정
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "sql_templates"

    # Elasticsearch 설정
    es_host: str = "localhost"
    es_port: int = 9200
    es_index_name: str = "sql_templates"

    # OpenAI API 설정
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"

    model_config = {
        "env_prefix": "TEXT2SQL_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

