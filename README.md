# Text2SQL 오프라인 파이프라인

Text2SQL 변환을 위한 오프라인 파이프라인 시스템입니다. Oracle IPA 로그에서 SQL 쿼리를 수집하여 정규화, 설명 생성, 벡터화를 수행하고 Milvus와 Elasticsearch에 인덱싱합니다.

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        오프라인 파이프라인                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
│  │  Log        │──▶│  Log        │──▶│  SQL        │──▶│ Description │ │
│  │  Collector  │   │  Filter     │   │  Normalizer │   │  Generator  │ │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘ │
│        │                                                     │         │
│        ▼                                                     ▼         │
│  ┌─────────────┐                                      ┌─────────────┐  │
│  │   Oracle    │                                      │   OpenAI    │  │
│  │   (IPA)     │                                      │   (LLM)     │  │
│  └─────────────┘                                      └─────────────┘  │
│                                                              │         │
│                    ┌─────────────────────────────────────────┘         │
│                    ▼                                                   │
│             ┌─────────────┐   ┌─────────────┐                          │
│             │  Vector     │   │   ES        │                          │
│             │  Indexer    │   │   Indexer   │                          │
│             └─────────────┘   └─────────────┘                          │
│                    │                 │                                 │
│                    ▼                 ▼                                 │
│             ┌─────────────┐   ┌─────────────┐                          │
│             │   Milvus    │   │Elasticsearch│                          │
│             └─────────────┘   └─────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📦 설치

### 1. venv 환경 생성 및 활성화

```bash
# venv 환경 생성
python3 -m venv venv

# 활성화 (macOS/Linux)
source venv/bin/activate

# 활성화 (Windows)
.\venv\Scripts\activate
```

### 2. 의존성 설치

```bash
# 개발 의존성 포함 설치 (권장)
pip install -e ".[dev]"

# 또는 requirements.txt로 설치
pip install -r requirements.txt
```

### 3. 환경 변수 설정

```bash
# .env.example을 복사하여 .env 파일 생성
cp .env.example .env

# .env 파일을 편집하여 실제 값 입력
vi .env
```

## 🚀 빠른 시작

### 데모 실행 (외부 인프라 불필요)

샘플 데이터로 파이프라인을 테스트합니다. Mock 컴포넌트를 사용하므로 외부 시스템 연결이 필요하지 않습니다.

```bash
# 데모 실행
python scripts/run_pipeline.py --demo

# 저장 없이 테스트 (dry-run)
python scripts/run_pipeline.py --demo --dry-run
```

**예시 출력:**
```
============================================================
🚀 Text2SQL 오프라인 파이프라인 데모 실행
============================================================

📂 샘플 데이터 로드: data/samples/sql_logs.json
   - 로드된 로그 수: 20

⚙️  파이프라인 실행 중...
  [VectorIndexer] 12개 템플릿 인덱싱 완료
  [ESIndexer] 12개 템플릿 인덱싱 완료

=== 파이프라인 실행 결과 ===
수집된 로그: 20건
필터링된 로그: 12건
정규화된 템플릿: 12건
인덱싱된 템플릿: 12건
에러 수: 0건
성공 여부: 성공
```

### 개별 컴포넌트 데모

각 컴포넌트를 개별적으로 테스트합니다.

```bash
# SQL 정규화기 데모
python scripts/demo_components.py normalizer

# 로그 필터 데모
python scripts/demo_components.py filter

# 용어 사전 데모
python scripts/demo_components.py glossary

# 모든 컴포넌트 데모
python scripts/demo_components.py all
```

## 🔧 실제 환경 실행

### 필수 인프라

실제 환경에서 파이프라인을 실행하려면 다음 시스템이 필요합니다:

1. **Oracle Database**: IPA 로그 테이블이 있는 Oracle DB
2. **Milvus**: 벡터 저장소 (버전 2.x 권장)
3. **Elasticsearch**: 전문 검색용 (버전 8.x 권장)
4. **OpenAI API Key**: SQL 설명 생성용

### 환경 변수 설정

`.env` 파일에 실제 연결 정보를 입력합니다:

```bash
# Oracle 설정
TEXT2SQL_ORACLE_HOST=your-oracle-host.example.com
TEXT2SQL_ORACLE_PORT=1521
TEXT2SQL_ORACLE_SERVICE_NAME=ORCL
TEXT2SQL_ORACLE_USER=your_user
TEXT2SQL_ORACLE_PASSWORD=your_password

# Milvus 설정
TEXT2SQL_MILVUS_HOST=your-milvus-host.example.com
TEXT2SQL_MILVUS_PORT=19530
TEXT2SQL_MILVUS_COLLECTION_NAME=sql_templates

# Elasticsearch 설정
TEXT2SQL_ES_HOST=your-es-host.example.com
TEXT2SQL_ES_PORT=9200
TEXT2SQL_ES_INDEX_NAME=sql_templates

# OpenAI 설정
TEXT2SQL_OPENAI_API_KEY=sk-your-actual-api-key
TEXT2SQL_OPENAI_MODEL=gpt-4o
```

### 파이프라인 실행

```bash
# 전체 파이프라인 실행
python scripts/run_pipeline.py

# 상위 100개 로그만 처리
python scripts/run_pipeline.py --limit 100

# dry-run 모드 (저장 없이 테스트)
python scripts/run_pipeline.py --dry-run
```

## 📊 샘플 데이터

### SQL 로그 샘플 (`data/samples/sql_logs.json`)

20개의 다양한 SQL 쿼리를 포함합니다:
- SELECT 쿼리 (단일 테이블, JOIN, 서브쿼리)
- 집계 함수 (COUNT, SUM, AVG)
- 윈도우 함수 (ROW_NUMBER, RANK)
- 필터링 대상 쿼리 (INSERT, UPDATE, DELETE, DDL, 시스템 테이블)

```json
{
  "sql_id": "SQL001",
  "sql_text": "SELECT customer_id, customer_name, email FROM customers WHERE customer_id = 12345",
  "exec_count": 15000,
  "error_count": 0,
  "collected_at": "2026-01-08T10:00:00",
  "schema_name": "SALES"
}
```

### 용어 사전 샘플 (`data/samples/glossary.csv`)

40개의 비즈니스 용어를 포함합니다:

| term | korean_name | description | category |
|------|-------------|-------------|----------|
| customer | 고객 | 서비스를 이용하거나 상품을 구매하는 개인 또는 법인 | 비즈니스 |
| order_date | 주문일자 | 주문이 생성된 날짜 | 일시 |
| total_amount | 총금액 | 주문의 총 결제 금액 | 금액 |

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest tests/ -v

# 특정 모듈 테스트
pytest tests/unit/offline/processor/test_sql_normalizer.py -v

# 커버리지 포함
pytest tests/ -v --cov=src/text2sql --cov-report=html
```

## 📁 프로젝트 구조

```
text2sql-pipeline/
├── data/
│   └── samples/
│       ├── sql_logs.json      # 샘플 SQL 로그
│       └── glossary.csv       # 샘플 용어 사전
├── scripts/
│   ├── run_pipeline.py        # 파이프라인 실행 스크립트
│   └── demo_components.py     # 컴포넌트 데모 스크립트
├── src/text2sql/
│   ├── core/                  # 데이터 모델 및 설정
│   │   ├── config.py          # Pydantic 설정
│   │   └── models.py          # 데이터클래스
│   ├── offline/               # 오프라인 파이프라인
│   │   ├── pipeline.py        # 오케스트레이터
│   │   ├── ingestor/          # 로그 수집/필터링
│   │   ├── processor/         # SQL 처리
│   │   ├── indexer/           # 인덱싱
│   │   └── schema/            # 스키마 빌더
│   └── adapters/              # 외부 시스템 어댑터
│       ├── database/          # Oracle
│       ├── vector_store/      # Milvus
│       ├── search/            # Elasticsearch
│       └── llm/               # OpenAI
├── tests/                     # 테스트 코드
├── .env.example               # 환경 변수 예시
├── pyproject.toml             # 프로젝트 설정
└── README.md                  # 이 파일
```

## 🔄 파이프라인 처리 흐름

1. **Log Collection**: Oracle IPA 테이블에서 SQL 로그 수집
2. **Log Filtering**: DML/DDL/시스템 쿼리 제외, SELECT만 통과
3. **SQL Normalization**: 리터럴 → placeholder 치환, 테이블/컬럼 추출
4. **Description Generation**: LLM을 통한 자연어 설명 생성
5. **Vector Indexing**: 임베딩 생성 후 Milvus에 저장
6. **ES Indexing**: BM25 검색용 Elasticsearch 인덱싱

## 📝 라이선스

MIT License
