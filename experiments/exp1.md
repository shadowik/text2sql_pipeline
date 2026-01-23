`datasketch` 라이브러리가 MinHash LSH를 독립적으로 구현해 LLM 훈련 데이터나 SQL 로그 같은 대규모 텍스트에서 효율적인 근사 중복 검출을 지원합니다. 이 라이브러리는 이전 첨부 파일에서 예시로 사용된 것처럼 shingling과 MinHash 시그니처 생성 후 LSH로 후보 쌍을 찾는 워크플로를 제공합니다. [milvus](https://milvus.io/ko/blog/minhash-lsh-in-milvus-the-secret-weapon-for-fighting-duplicates-in-llm-training-data.md)

## 주요 독립 라이브러리
- **datasketch**: MinHash와 LSH를 위한 표준 라이브러리. `MinHash` 객체로 문서 시그니처 생성 후 `MinHashLSH` 인덱스에 추가해 쿼리합니다. 설치: `pip install datasketch`.
- **spark-dedup** 또는 **dedupe**: 대규모 데이터셋용, probabilistic record linkage 지원. LLM 데이터보다는 구조화 데이터에 강함.
- **rapids-cudf** (GPU 가속): NVIDIA RAPIDS로 MinHash LSH를 스케일링한 cuDF, 수십억 문서 처리에 적합하지만 GPU 필요. [blog.naver](https://blog.naver.com/dsz08082/222564464414)

## Text-to-SQL 프로젝트 적용 예시
Text-to-SQL 파이프라인에서 SQL 로그 중복 체크 시, datasketch로 normalized SQL 템플릿의 MinHash 생성 후 LSH 인덱스 검색(threshold 0.95)을 Pre-Dedup 단계에 삽입하면 Milvus 없이도 upsert 비용을 줄일 수 있습니다. 코드 예시: [perplexity](https://www.perplexity.ai/search/93f475d8-7448-4141-8aa5-152473f2a0b7)

```python
from datasketch import MinHash, MinHashLSH
lsh = MinHashLSH(threshold=0.9, num_perm=128)
for doc_id, text in documents.items():
    m = MinHash(num_perm=128)
    for shingle in set(text.lower().split()):  # 또는 k-shingles
        m.update(shingle.encode('utf-8'))
    lsh.insert(doc_id, m)
# 쿼리: candidates = lsh.query(query_minhash)
```