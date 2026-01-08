from pymilvus import utility, connections

# Milvus 연결
connections.connect(host="localhost", port="19530")

# 컬렉션 삭제 (컬렉션 이름은 설정에서 확인)
utility.drop_collection("your_collection_name")