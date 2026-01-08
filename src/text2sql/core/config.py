"""애플리케이션 설정 모듈."""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정."""

    # Oracle 데이터베이스 설정
    oracle_host: str = "localhost"
    oracle_port: int = 1521
    oracle_service_name: str = "FREEPDB1"
    oracle_user: Optional[str] = "text2sql"
    oracle_password: Optional[str] = "text2sql123"

    # Milvus 벡터 저장소 설정
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "sql_templates"

    # Elasticsearch 설정
    es_host: str = "localhost"
    es_port: int = 9200
    es_index_name: str = "sql_templates"

    # LLM API 설정 (OpenAI 호환 - LM Studio 등)
    llm_base_url: str = "http://192.168.0.99:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = "qwen3-4b-instruct-2507-mlx"

    # 임베딩 API 설정
    embedding_base_url: str = "http://192.168.0.99:1234/v1"
    embedding_api_key: str = "lm-studio"
    embedding_model: str = "text-embedding-qwen3-embedding-0.6b"

    model_config = {
        "env_prefix": "TEXT2SQL_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

