# Docker Infrastructure

Text2SQL Agent를 위한 Docker 인프라 구성입니다.

> **중요**: Milvus 2.6+ 버전은 **MinHash LSH 인덱싱**을 지원하며, 이는 SQL 템플릿 중복 제거(Dedup) 파이프라인의 핵심 기능입니다. 자세한 내용은 `docs/pipeline_design_draft.md`를 참조하세요.

## 서비스 구성

| 서비스 | 버전 | 포트 | 설명 |
|--------|------|------|------|
| Milvus | **2.6.9** | 19530 | 벡터 데이터베이스 (MinHash LSH 지원) |
| Attu | **2.6.4** | 8000 | Milvus 웹 GUI |
| Oracle Free | 23.5 | 1521 | 관계형 데이터베이스 (SQL 로그/메타 저장) |
| MinIO | 2024-12-18 | 9000, 9001 | Milvus 오브젝트 스토리지 |
| etcd | 3.5.0 | - | Milvus 메타데이터 저장 |

## 시작 방법

```bash
# 전체 서비스 시작
docker-compose up -d

# 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs -f [service_name]

# 전체 서비스 중지
docker-compose down

# 볼륨 포함 삭제 (주의: 데이터 삭제됨)
docker-compose down -v
```

---

## 📊 데이터 확인용 웹 UI 접속 정보

| 서비스 | 웹 UI URL | 설명 |
|--------|-----------|------|
| **Milvus** | http://localhost:8000 | Attu - Vector DB GUI |
| **MinIO** | http://localhost:9001 | MinIO Console - Object Storage |

---

## 🔍 각 서비스별 데이터 확인 방법

### 1. Milvus (Vector Database) - Attu GUI

**URL:** http://localhost:8000

1. 브라우저에서 접속
2. 자동으로 Milvus에 연결됨 (설정된 `MILVUS_URL: milvus:19530`)
3. 왼쪽 메뉴에서 확인 가능:
   - **Collections**: 생성된 컬렉션 목록 및 스키마 확인
   - **Data**: 컬렉션 내 벡터 데이터 검색 및 조회
   - **Search**: 벡터 유사도 검색 테스트
   - **Index**: 인덱스 상태 확인

**CLI로 확인:**
```bash
# Milvus 연결 상태
curl http://localhost:9091/healthz

# Python으로 확인
python -c "
from pymilvus import connections, utility
connections.connect('default', host='localhost', port='19530')
print('Collections:', utility.list_collections())
"
```

---

| 항목 | 값 |
|------|-----|
| Host | `oracle` (Docker 네트워크 내) 또는 `host.docker.internal` |
| Port | `1521` |
| Database (SID) | `FREEPDB1` |
| User | `text2sql` |
| Password | `text2sql123` |

5. **Test Connection** → **Save**

**데이터 확인:**
- 왼쪽 패널에서 연결 확장 → 스키마 → 테이블 선택
- 테이블 우클릭 → **View Data** 로 데이터 조회
- SQL Editor에서 직접 쿼리 실행 가능

**CLI로 확인 (Docker exec):**
```bash
# Oracle 컨테이너에 접속하여 SQL*Plus 실행
docker exec -it oracle-xe sqlplus text2sql/text2sql123@localhost:1521/FREEPDB1

# SQL*Plus 내에서
SQL> SELECT table_name FROM user_tables;
SQL> SELECT COUNT(*) FROM your_table_name;
SQL> EXIT;
```

---

### 4. MinIO (Object Storage) - MinIO Console

**URL:** http://localhost:9001

**로그인 정보:**
- Access Key: `minioadmin`
- Secret Key: `minioadmin`

**확인 방법:**
1. 브라우저에서 접속 후 로그인
2. **Buckets** 메뉴에서 생성된 버킷 목록 확인
3. 버킷 클릭 → 저장된 파일/객체 목록 조회

---

## 헬스체크

```bash
# 전체 서비스 상태
docker-compose ps

# Milvus
curl http://localhost:9091/healthz

# Oracle (컨테이너 내부)
docker exec oracle-xe healthcheck.sh
```

---

## 접속 정보 요약

### Milvus
- Host: `localhost`
- Port: `19530`
- **Attu GUI:** http://localhost:8000

### Oracle
- Host: `localhost`
- Port: `1521`
- SID: `FREEPDB1`
- System Password: `oracle123`
- App User: `text2sql` / `text2sql123`

### MinIO (Milvus Storage)
- **Console:** http://localhost:9001
- Access Key: `minioadmin`
- Secret Key: `minioadmin`

---

## 트러블슈팅

```bash
docker logs oracle-xe
```

### 서비스 개별 재시작
```bash
docker-compose restart [service_name]
```
