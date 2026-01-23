#!/usr/bin/env python3
"""
SQL 유사도 검증 테스트 스크립트

datasketch 라이브러리의 MinHash LSH를 사용하여 SQL 로그의 유사 여부를 검증합니다.
"""

from datasketch import MinHash, MinHashLSH
from typing import List, Dict, Tuple
import re


def normalize_sql(sql: str) -> str:
    """SQL을 정규화하여 비교 가능한 형태로 변환합니다."""
    sql = sql.upper()
    sql = re.sub(r'\s+', ' ', sql)  # 연속 공백 제거
    sql = re.sub(r"'[^']*'", "'?'", sql)  # 문자열 리터럴 치환
    sql = re.sub(r'\b\d+\b', '?', sql)  # 숫자 리터럴 치환
    sql = sql.strip()
    return sql


def create_minhash(text: str, num_perm: int = 128, k: int = 3) -> MinHash:
    """텍스트에서 k-shingle 기반 MinHash를 생성합니다."""
    m = MinHash(num_perm=num_perm)
    # k-shingles (연속된 k개 문자) 생성
    text = text.lower()
    for i in range(len(text) - k + 1):
        shingle = text[i:i + k]
        m.update(shingle.encode('utf-8'))
    return m


def calculate_jaccard_similarity(mh1: MinHash, mh2: MinHash) -> float:
    """두 MinHash 간의 Jaccard 유사도를 계산합니다."""
    return mh1.jaccard(mh2)


def print_separator(char: str = "=", length: int = 80):
    """구분선을 출력합니다."""
    print(char * length)


def print_header(title: str):
    """섹션 헤더를 출력합니다."""
    print_separator()
    print(f" {title}")
    print_separator()


def print_sql_pair(idx1: int, idx2: int, sql1: str, sql2: str, similarity: float, is_similar: bool):
    """SQL 쌍과 유사도 정보를 출력합니다."""
    status = "✅ 유사" if is_similar else "❌ 다름"
    print(f"\n[비교] SQL #{idx1} vs SQL #{idx2}")
    print(f"  유사도: {similarity:.4f} ({similarity * 100:.2f}%)")
    print(f"  판정: {status}")
    print(f"  SQL #{idx1}: {sql1[:60]}..." if len(sql1) > 60 else f"  SQL #{idx1}: {sql1}")
    print(f"  SQL #{idx2}: {sql2[:60]}..." if len(sql2) > 60 else f"  SQL #{idx2}: {sql2}")


def main():
    # 테스트용 SQL 로그 데이터 생성
    # 그룹 A: 사용자 조회 관련 (유사한 SQL들)
    # 그룹 B: 주문 조회 관련 (유사한 SQL들)
    # 그룹 C: 상품 조회 관련 (유사한 SQL들)
    # 그룹 D: 완전히 다른 SQL들
    
    sql_logs = {
        # 그룹 A: 사용자 조회 (유사한 변형들)
        "A1": "SELECT l.lot_id, l.fab_id, l.hold_code, l.hold_reason, l.create_dt, e.eqp_name FROM MES_BIZ_LOTHOLD_INF_M11 l LEFT JOIN MES_EQP_MST_M11 e ON l.eqp_id = e.eqp_id WHERE l.hold_code IS NOT NULL AND l.create_dt >= TO_DATE('2026-01-01', 'YYYY-MM-DD') ORDER BY l.create_dt DESC",
        "A2": "SELECT l.lot_id, l.fab_id, l.hold_code, l.hold_reason, l.create_dt, e.eqp_name FROM MES_BIZ_LOTHOLD_INF_M11 l LEFT JOIN MES_EQP_MST_M11 e ON l.eqp_id = e.eqp_id WHERE l.hold_code IS NOT NULL AND l.create_dt >= TO_DATE('2026-01-01', 'YYYY-MM-DD') ORDER BY l.create_dt ASC",
        "A3": "SELECT l.lot_id, l.fab_id, l.hold_code, l.hold_reason, l.create_dt, e.eqp_name FROM MES_BIZ_LOTHOLD_INF_M12 l LEFT JOIN MES_EQP_MST_M12 e ON l.eqp_id = e.eqp_id WHERE l.hold_code IS NOT NULL AND l.create_dt >= TO_DATE('2026-01-01', 'YYYY-MM-DD') ORDER BY l.create_dt DESC",
        "A4": "SELECT user_id, user_name, email, phone FROM users WHERE user_id = 101",
        
        # 그룹 B: 주문 조회 (유사한 변형들)
        "B1": "SELECT order_id, customer_id, total_amount FROM orders WHERE order_date >= '2024-01-01'",
        "B2": "SELECT order_id, customer_id, total_amount FROM orders WHERE order_date >= '2024-06-01'",
        "B3": "SELECT order_id, customer_id, total_amount, status FROM orders WHERE order_date >= '2024-03-15'",
        
        # 그룹 C: 상품 조회 (유사한 변형들)
        "C1": "SELECT product_id, product_name, price, category FROM products WHERE category = 'electronics'",
        "C2": "SELECT product_id, product_name, price, category FROM products WHERE category = 'clothing'",
        "C3": "SELECT product_id, product_name, price FROM products WHERE category = 'books' AND price < 50",
        
        # 그룹 D: 완전히 다른 SQL들
        "D1": "INSERT INTO audit_logs (action, timestamp, user_id) VALUES ('login', NOW(), 1)",
        "D2": "DELETE FROM sessions WHERE last_activity < DATE_SUB(NOW(), INTERVAL 30 DAY)",
        "D3": "UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 999",
    }
    
    # 설정
    THRESHOLD = 0.5  # 유사도 임계값
    NUM_PERM = 128   # MinHash 순열 수
    K_SHINGLE = 3    # k-shingle 크기
    
    print_header("SQL 유사도 검증 테스트")
    print(f"\n설정:")
    print(f"  - 유사도 임계값 (threshold): {THRESHOLD}")
    print(f"  - MinHash 순열 수 (num_perm): {NUM_PERM}")
    print(f"  - k-shingle 크기: {K_SHINGLE}")
    print(f"  - 총 SQL 수: {len(sql_logs)}")
    
    # 1. SQL 정규화 및 MinHash 생성
    print_header("1단계: SQL 정규화 및 MinHash 생성")
    
    normalized_sqls: Dict[str, str] = {}
    minhashes: Dict[str, MinHash] = {}
    
    for doc_id, sql in sql_logs.items():
        normalized = normalize_sql(sql)
        normalized_sqls[doc_id] = normalized
        minhashes[doc_id] = create_minhash(normalized, num_perm=NUM_PERM, k=K_SHINGLE)
        print(f"\n[{doc_id}] 원본: {sql[:50]}...")
        print(f"      정규화: {normalized[:50]}...")
    
    # 2. LSH 인덱스 생성 및 삽입
    print_header("2단계: MinHashLSH 인덱스 생성")
    
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    for doc_id, mh in minhashes.items():
        lsh.insert(doc_id, mh)
        print(f"  인덱스에 추가됨: {doc_id}")
    
    # 3. 각 SQL에 대해 유사한 SQL 쿼리
    print_header("3단계: 유사 SQL 검색 결과")
    
    similar_groups: Dict[str, List[str]] = {}
    for doc_id, mh in minhashes.items():
        candidates = lsh.query(mh)
        # 자기 자신 제외
        similar = [c for c in candidates if c != doc_id]
        similar_groups[doc_id] = similar
        
        print(f"\n[{doc_id}] 유사한 SQL 후보: {similar if similar else '없음'}")
    
    # 4. 상세 유사도 행렬 출력
    print_header("4단계: 상세 유사도 분석")
    
    doc_ids = list(sql_logs.keys())
    
    # 헤더 출력
    print("\n" + " " * 6, end="")
    for doc_id in doc_ids:
        print(f"{doc_id:>6}", end="")
    print()
    
    # 유사도 행렬 출력
    for i, id1 in enumerate(doc_ids):
        print(f"{id1:>6}", end="")
        for j, id2 in enumerate(doc_ids):
            if i == j:
                print(f"{'1.00':>6}", end="")
            elif j > i:
                sim = calculate_jaccard_similarity(minhashes[id1], minhashes[id2])
                print(f"{sim:>6.2f}", end="")
            else:
                print(f"{'':>6}", end="")
        print()
    
    # 5. 그룹별 분석 결과
    print_header("5단계: 그룹별 유사도 분석")
    
    groups = {
        "A": ["A1", "A2", "A3", "A4"],
        "B": ["B1", "B2", "B3"],
        "C": ["C1", "C2", "C3"],
        "D": ["D1", "D2", "D3"],
    }
    
    for group_name, members in groups.items():
        print(f"\n=== 그룹 {group_name} (예상: 같은 그룹은 유사해야 함) ===")
        
        # 그룹 내 유사도
        intra_similarities = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                sim = calculate_jaccard_similarity(minhashes[members[i]], minhashes[members[j]])
                intra_similarities.append(sim)
                status = "✅" if sim >= THRESHOLD else "⚠️"
                print(f"  {status} {members[i]} vs {members[j]}: {sim:.4f}")
        
        if intra_similarities:
            avg_sim = sum(intra_similarities) / len(intra_similarities)
            print(f"  📊 그룹 내 평균 유사도: {avg_sim:.4f}")
    
    # 6. 그룹 간 유사도 분석
    print_header("6단계: 그룹 간 유사도 (다른 그룹은 낮아야 함)")
    
    inter_group_sims = []
    for g1_name, g1_members in groups.items():
        for g2_name, g2_members in groups.items():
            if g1_name >= g2_name:
                continue
            
            sims = []
            for m1 in g1_members:
                for m2 in g2_members:
                    sim = calculate_jaccard_similarity(minhashes[m1], minhashes[m2])
                    sims.append(sim)
            
            avg_sim = sum(sims) / len(sims) if sims else 0
            inter_group_sims.append(avg_sim)
            status = "✅" if avg_sim < THRESHOLD else "⚠️"
            print(f"  {status} 그룹 {g1_name} vs 그룹 {g2_name}: 평균 {avg_sim:.4f}")
    
    # 7. 최종 요약
    print_header("📊 최종 요약")
    
    # 그룹 내 평균 계산
    all_intra_sims = []
    for group_name, members in groups.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                sim = calculate_jaccard_similarity(minhashes[members[i]], minhashes[members[j]])
                all_intra_sims.append(sim)
    
    avg_intra = sum(all_intra_sims) / len(all_intra_sims) if all_intra_sims else 0
    avg_inter = sum(inter_group_sims) / len(inter_group_sims) if inter_group_sims else 0
    
    print(f"\n  총 SQL 수: {len(sql_logs)}")
    print(f"  유사도 임계값: {THRESHOLD}")
    print(f"  그룹 내 평균 유사도: {avg_intra:.4f}")
    print(f"  그룹 간 평균 유사도: {avg_inter:.4f}")
    print(f"  분리도 (그룹 내 - 그룹 간): {avg_intra - avg_inter:.4f}")
    
    if avg_intra > avg_inter:
        print(f"\n  ✅ 결론: MinHash LSH가 유사 SQL을 잘 구분하고 있습니다!")
    else:
        print(f"\n  ⚠️ 결론: 파라미터 조정이 필요할 수 있습니다.")
    
    print_separator()


if __name__ == "__main__":
    main()
