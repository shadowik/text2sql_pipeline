"""Oracle 데이터베이스 어댑터."""

from typing import Any

import oracledb

from text2sql.core.config import Settings


class OracleAdapter:
    """Oracle 데이터베이스 어댑터."""

    def __init__(self, settings: Settings) -> None:
        """어댑터 초기화.

        Args:
            settings: 애플리케이션 설정
        """
        self._settings = settings
        self._connection: Any = None
        self._pool: Any = None

    def connect(self) -> Any:
        """Oracle 데이터베이스에 연결.

        Returns:
            데이터베이스 연결 객체
        """
        dsn = f"{self._settings.oracle_host}:{self._settings.oracle_port}/{self._settings.oracle_service_name}"
        self._connection = oracledb.connect(
            user=self._settings.oracle_user,
            password=self._settings.oracle_password,
            dsn=dsn,
        )
        return self._connection

    def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[dict[str, Any]]:
        """SELECT 쿼리를 실행하고 결과를 반환.

        Args:
            query: 실행할 SQL 쿼리
            timeout: 쿼리 타임아웃 (초 단위)

        Returns:
            딕셔너리 리스트 형태의 쿼리 결과
        """
        if self._connection is None:
            raise RuntimeError("연결이 설정되지 않았습니다. connect()를 먼저 호출하세요.")

        # 타임아웃 설정 (밀리초 단위로 변환)
        if timeout is not None:
            self._connection.call_timeout = timeout * 1000

        with self._connection.cursor() as cursor:
            cursor.execute(query)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]

    def create_pool(self, min_connections: int = 2, max_connections: int = 10) -> Any:
        """연결 풀 생성.

        Args:
            min_connections: 최소 연결 수
            max_connections: 최대 연결 수

        Returns:
            연결 풀 객체
        """
        dsn = f"{self._settings.oracle_host}:{self._settings.oracle_port}/{self._settings.oracle_service_name}"
        self._pool = oracledb.create_pool(
            user=self._settings.oracle_user,
            password=self._settings.oracle_password,
            dsn=dsn,
            min=min_connections,
            max=max_connections,
        )
        return self._pool

    def acquire_connection(self) -> Any:
        """풀에서 연결을 가져옴.

        Returns:
            데이터베이스 연결 객체
        """
        if self._pool is None:
            raise RuntimeError("연결 풀이 생성되지 않았습니다. create_pool()을 먼저 호출하세요.")
        return self._pool.acquire()

    def release_connection(self, connection: Any) -> None:
        """연결을 풀에 반환.

        Args:
            connection: 반환할 연결 객체
        """
        if self._pool is None:
            raise RuntimeError("연결 풀이 생성되지 않았습니다.")
        self._pool.release(connection)

