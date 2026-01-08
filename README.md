# Text2SQL 오프라인 파이프라인

Text2SQL 변환을 위한 오프라인 파이프라인 시스템입니다.

## 설치

```bash
# venv 환경 생성
python3 -m venv venv
source venv/bin/activate

# 개발 의존성 포함 설치
pip install -e ".[dev]"
```

## 테스트 실행

```bash
# 전체 테스트
pytest tests/ -v

# 커버리지 포함
pytest tests/ -v --cov=src/text2sql --cov-report=html
```

## 프로젝트 구조

```
src/text2sql/
├── core/          # 데이터 모델 및 설정
├── offline/       # 오프라인 파이프라인
└── adapters/      # 외부 시스템 어댑터
```

