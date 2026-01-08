"""Configuration - 환경별 설정 관리.

pydantic-settings를 사용하여 환경 변수 및 .env 파일에서 설정을 로드합니다.
"""

from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """실행 환경."""

    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class MilvusSettings(BaseSettings):
    """Milvus 연결 설정."""

    host: str = Field(default="localhost", description="Milvus 호스트")
    port: int = Field(default=19530, description="Milvus 포트")
    collection_name: str = Field(
        default="sql_templates", description="SQL 템플릿 컬렉션명"
    )
    embedding_dim: int = Field(default=1536, description="임베딩 벡터 차원")

    model_config = SettingsConfigDict(env_prefix="MILVUS_")


class ElasticsearchSettings(BaseSettings):
    """Elasticsearch 연결 설정."""

    host: str = Field(default="localhost", description="ES 호스트")
    port: int = Field(default=9200, description="ES 포트")
    scheme: str = Field(default="http", description="프로토콜")
    index_prefix: str = Field(default="text2sql", description="인덱스 접두사")

    @property
    def url(self) -> str:
        """ES URL을 반환합니다."""
        return f"{self.scheme}://{self.host}:{self.port}"

    model_config = SettingsConfigDict(env_prefix="ES_")


class OracleSettings(BaseSettings):
    """Oracle 연결 설정."""

    host: str = Field(default="localhost", description="Oracle 호스트")
    port: int = Field(default=1521, description="Oracle 포트")
    service_name: str = Field(default="FREEPDB1", description="서비스명")
    user: str = Field(default="text2sql", description="사용자명")
    password: str = Field(default="text2sql123", description="비밀번호")

    @property
    def dsn(self) -> str:
        """Oracle DSN을 반환합니다."""
        return f"{self.host}:{self.port}/{self.service_name}"

    model_config = SettingsConfigDict(env_prefix="ORACLE_")


class LLMSettings(BaseSettings):
    """LLM API 설정."""

    base_url: str = Field(
        default="http://192.168.0.99:1234/v1",
        description="OpenAI 호환 API 기본 URL",
    )
    api_key: str = Field(
        default="lm-studio",
        description="API 키 (LM Studio는 더미값 사용)",
    )
    model: str = Field(
        default="qwen3-4b-instruct-2507-mlx",
        description="사용할 모델명",
    )
    temperature: float = Field(default=0.1, description="생성 온도")
    max_tokens: int = Field(default=4096, description="최대 토큰 수")
    timeout: float = Field(default=60.0, description="요청 타임아웃 (초)")

    model_config = SettingsConfigDict(env_prefix="LLM_")


class EmbeddingSettings(BaseSettings):
    """임베딩 서비스 설정."""

    base_url: str = Field(
        default="http://192.168.0.99:1234/v1",
        description="임베딩 API 기본 URL",
    )
    api_key: str = Field(default="lm-studio", description="API 키")
    model: str = Field(
        default="text-embedding-nomic-embed-text-v1.5",
        description="임베딩 모델명",
    )

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")


class AgentSettings(BaseSettings):
    """에이전트 동작 설정."""

    similarity_threshold: float = Field(
        default=0.8, description="1단계 유사도 임계값"
    )
    top_k_templates: int = Field(default=5, description="검색할 템플릿 개수")
    sql_row_limit: int = Field(default=1000, description="SQL 결과 row 상한")
    sql_timeout_sec: int = Field(default=30, description="SQL 실행 timeout")
    session_ttl_min: int = Field(default=60, description="세션 만료 시간")
    max_clarify_turns: int = Field(default=3, description="Clarifying 질문 최대 횟수")
    max_self_check_loops: int = Field(default=3, description="Self-check 최대 반복 횟수")

    model_config = SettingsConfigDict(env_prefix="AGENT_")


class A2ASettings(BaseSettings):
    """A2A 프로토콜 설정."""

    enabled: bool = Field(default=True, description="A2A 엔드포인트 활성화 여부")
    rate_limit: int = Field(default=100, description="분당 최대 요청 수")
    task_timeout_sec: int = Field(default=60, description="Task 처리 타임아웃")
    max_history: int = Field(default=100, description="유지할 최대 Task 히스토리 수")

    model_config = SettingsConfigDict(env_prefix="A2A_")


class Settings(BaseSettings):
    """전체 애플리케이션 설정."""

    env: Environment = Field(default=Environment.DEV, description="실행 환경")
    debug: bool = Field(default=True, description="디버그 모드")
    app_name: str = Field(default="Text2SQL Agent", description="앱 이름")
    api_prefix: str = Field(default="/api/v1", description="API 경로 접두사")

    # 하위 설정
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    elasticsearch: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
    oracle: OracleSettings = Field(default_factory=OracleSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    a2a: A2ASettings = Field(default_factory=A2ASettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """설정 싱글톤 인스턴스를 반환합니다."""
    return Settings()


# 편의를 위한 전역 설정 접근
settings = get_settings()

