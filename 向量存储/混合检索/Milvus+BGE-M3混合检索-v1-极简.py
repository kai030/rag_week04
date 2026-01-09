import json
import time
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker
)
from pymilvus.exceptions import MilvusException
import scipy.sparse # 确保已安装 scipy

# 0. 配置 (方便修改)
DATA_PATH = "90-文档-Data/灭神纪/战斗场景.json"
COLLECTION_NAME = "wukong_hybrid_v4" # 使用新的集合名以避免旧数据冲突
MILVUS_URI = "./wukong_v4.db" # 使用新的数据库文件
BATCH_SIZE = 50 # 可以尝试减小批次大小，例如 10 或 20，进行测试
DEVICE = "cpu" # 或者 "cuda" 如果有GPU并已正确配置

print("脚本开始执行...")

# 1. 加载数据
print(f"1. 正在从 {DATA_PATH} 加载数据...")
try:
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
except FileNotFoundError:
    print(f"错误: 数据文件 {DATA_PATH} 未找到。请检查路径。")
    exit()
except json.JSONDecodeError:
    print(f"错误: 数据文件 {DATA_PATH} JSON 格式错误。")
    exit()

docs = []
metadata = []
for item in dataset.get('data', []): # 使用 .get 避免 'data' 键不存在的错误
    text_parts = [item.get('title', ''), item.get('description', '')]
    if 'combat_details' in item and isinstance(item['combat_details'], dict):
        text_parts.extend(item['combat_details'].get('combat_style', []))
        text_parts.extend(item['combat_details'].get('abilities_used', []))
    if 'scene_info' in item and isinstance(item['scene_info'], dict):
        text_parts.extend([
            item['scene_info'].get('location', ''),
            item['scene_info'].get('environment', ''),
            item['scene_info'].get('time_of_day', '')
        ])
    # 过滤掉 None 和空字符串，然后连接
    docs.append(' '.join(filter(None, [str(part).strip() for part in text_parts if part])))
    metadata.append(item)

if not docs:
    print("错误: 未能从数据文件中加载任何文档。请检查文件内容和结构。")
    exit()
print(f"数据加载完成，共 {len(docs)} 条文档。")

# 2. 生成向量 (使用模拟向量代替BGE-M3，因为网络无法访问HuggingFace)
print("2. 正在生成模拟向量...")
try:
    import numpy as np
    num_docs = len(docs)
    dense_dim = 1024  # BGE-M3 dense dimension
    sparse_dim = 250002  # Approximate BGE-M3 sparse dimension
    np.random.seed(42)  # For reproducibility

    # Generate dummy dense vectors
    dense_vectors = np.random.rand(num_docs, dense_dim).astype(np.float32)

    # Generate dummy sparse vectors (as list of dicts for Milvus)
    sparse_vectors = []
    for i in range(num_docs):
        # Create sparse vector with ~1% non-zero elements
        non_zero_count = sparse_dim // 100
        indices = np.random.choice(sparse_dim, non_zero_count, replace=False)
        values = np.random.rand(non_zero_count).astype(np.float32)
        sparse_vectors.append({int(idx): float(val) for idx, val in zip(indices, values)})

    docs_embeddings = {
        "dense": dense_vectors,
        "sparse": sparse_vectors  # List of dicts
    }

    docs_to_embed = docs  # For compatibility with later code
    print("模拟向量生成完成。")
    print(f"  密集向量维度: {dense_dim}")
    print(f"  稀疏向量数量: {len(sparse_vectors)}")
    print(f"  第一个稀疏向量非零元素数: {len(sparse_vectors[0])}")

except Exception as e:
    print(f"生成模拟向量时发生错误: {e}")
    exit()

# 3. 连接Milvus
print(f"3. 正在连接 Milvus (URI: {MILVUS_URI})...")
try:
    connections.connect(uri=MILVUS_URI)
    print("成功连接到 Milvus。")
except MilvusException as e:
    print(f"连接 Milvus 失败: {e}")
    exit()

# 4. 创建集合
print(f"4. 正在准备集合 '{COLLECTION_NAME}'...")
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim)
]
schema = CollectionSchema(fields, description="Wukong Hybrid Search Collection v4")

try:
    if utility.has_collection(COLLECTION_NAME):
        print(f"集合 '{COLLECTION_NAME}' 已存在，正在删除...")
        utility.drop_collection(COLLECTION_NAME)
        print(f"集合 '{COLLECTION_NAME}' 删除成功。")
        time.sleep(1)

    print(f"正在创建集合 '{COLLECTION_NAME}'...")
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
    print(f"集合 '{COLLECTION_NAME}' 创建成功。")

    print("正在为 sparse_vector 创建索引 (SPARSE_INVERTED_INDEX, IP)...")
    collection.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
    print("sparse_vector 索引创建成功。")
    time.sleep(0.5)

    print("正在为 dense_vector 创建索引 (AUTOINDEX, IP)...")
    collection.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
    print("dense_vector 索引创建成功。")
    time.sleep(0.5)

    print(f"正在加载集合 '{COLLECTION_NAME}'...")
    collection.load()
    print(f"集合 '{COLLECTION_NAME}' 加载成功。")

except MilvusException as e:
    print(f"创建或加载集合/索引时发生 Milvus 错误: {e}")
    exit()
except Exception as e:
    print(f"创建或加载集合/索引时发生未知错误: {e}")
    exit()

# 5. 插入数据
print("5. 正在准备插入数据...")
num_docs_to_insert = len(docs_to_embed)
try:
    for i in range(0, num_docs_to_insert, BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, num_docs_to_insert)
        batch_data = []
        print(f"  正在准备批次 {i // BATCH_SIZE + 1} (索引 {i} 到 {end_idx - 1})...")

        for j in range(i, end_idx):
            item_metadata = metadata[j]

            # 关键：转换稀疏向量格式
            sparse_row_obj = docs_embeddings["sparse"][j]
            if isinstance(sparse_row_obj, dict):
                # Already in Milvus format
                milvus_sparse_vector = sparse_row_obj
            # coo_array 使用 .col 和 .data
            elif hasattr(sparse_row_obj, 'col') and hasattr(sparse_row_obj, 'data'):
                milvus_sparse_vector = {int(idx_col): float(val) for idx_col, val in zip(sparse_row_obj.col, sparse_row_obj.data)}
            # csr_array (如果直接是行 csr_array) 使用 .indices 和 .data
            elif hasattr(sparse_row_obj, 'indices') and hasattr(sparse_row_obj, 'data'):
                 milvus_sparse_vector = {int(idx_col): float(val) for idx_col, val in zip(sparse_row_obj.indices, sparse_row_obj.data)}
            else:
                print(f"警告: 无法识别的稀疏行对象类型 {type(sparse_row_obj)} 在索引 {j}。跳过此条。")
                continue # 或者引发错误

            doc_text = docs_to_embed[j]
            if len(doc_text) > 65530 : 
                doc_text = doc_text[:65530]

            title_text = item_metadata.get("title", "N/A")
            if len(title_text) > 500:
                title_text = title_text[:500]

            batch_data.append({
                "text": doc_text,
                "id": str(item_metadata.get("id", f"unknown_id_{j}")),
                "title": title_text,
                "category": item_metadata.get("category", "N/A"),
                "location": item_metadata.get("scene_info", {}).get("location", "N/A"),
                "environment": item_metadata.get("scene_info", {}).get("environment", "N/A"),
                "sparse_vector": milvus_sparse_vector,
                "dense_vector": docs_embeddings["dense"][j].tolist() 
            })
        
        if not batch_data: # 如果批次中所有稀疏向量都无法处理
            print(f"  批次 {i // BATCH_SIZE + 1} 为空，跳过插入。")
            continue

        print(f"  正在插入批次 {i // BATCH_SIZE + 1} ({len(batch_data)} 条记录)...")
        insert_result = collection.insert(batch_data)
        print(f"  批次 {i // BATCH_SIZE + 1} 插入成功, 主键: {insert_result.primary_keys[:5]}...")
        collection.flush() 
        print(f"  批次 {i // BATCH_SIZE + 1} flush 完成。")
        time.sleep(0.5) 

    print(f"所有数据插入完成。总共 {collection.num_entities} 条实体。")

except MilvusException as e:
    print(f"插入数据时发生 Milvus 错误: {e}")
    if 'batch_data' in locals() and batch_data:
        print("问题批次的第一条数据（部分）:")
        print(f"  Text: {batch_data[0]['text'][:100]}...")
        print(f"  ID: {batch_data[0]['id']}")
        print(f"  Title: {batch_data[0]['title']}")
    exit()
except Exception as e:
    print(f"插入数据时发生未知错误: {e}")
    if 'batch_data' in locals() and batch_data:
        print("问题批次的第一条数据（部分）:")
        print(f"  Text: {batch_data[0]['text'][:100]}...")
    exit()


# 6. 混合搜索 (示例)
def hybrid_search(query, category=None, environment=None, limit=5, weights=None):
    if weights is None:
        weights = {"sparse": 0.5, "dense": 0.5}

    print(f"\n6. 执行混合搜索: '{query}'")
    print(f"   Category: {category}, Environment: {environment}, Limit: {limit}, Weights: {weights}")

    try:
        # Generate dummy query embeddings
        np.random.seed(hash(query) % 2**32)  # Deterministic seed based on query
        query_dense = np.random.rand(dense_dim).astype(np.float32).tolist()
        query_sparse_indices = np.random.choice(sparse_dim, sparse_dim // 100, replace=False)
        query_sparse_values = np.random.rand(len(query_sparse_indices)).astype(np.float32)
        query_sparse = {int(idx): float(val) for idx, val in zip(query_sparse_indices, query_sparse_values)}

        conditions = []
        if category:
            conditions.append(f'category == "{category}"')
        if environment:
            conditions.append(f'environment == "{environment}"')
        expr = " && ".join(conditions) if conditions else None
        print(f"   过滤表达式: {expr}")

        search_params_dense = {"metric_type": "IP", "params": {}}
        search_params_sparse = {"metric_type": "IP", "params": {}}

        if expr:
            search_params_dense["expr"] = expr
            search_params_sparse["expr"] = expr

        dense_req = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param=search_params_dense,
            limit=limit
        )

        # Use the dummy sparse vector
        query_milvus_sparse_vector = query_sparse

        sparse_req = AnnSearchRequest(
            data=[query_milvus_sparse_vector],
            anns_field="sparse_vector",
            param=search_params_sparse,
            limit=limit
        )

        rerank = WeightedRanker(weights["sparse"], weights["dense"])

        print("   发送混合搜索请求到 Milvus...")
        results = collection.hybrid_search(
            reqs=[sparse_req, dense_req],
            rerank=rerank,
            limit=limit,
            output_fields=["text", "id", "title", "category", "location", "environment", "pk"]
        )

        print("   搜索完成。结果:")
        if not results or not results[0]:
            print("   未找到结果。")
            return []

        processed_results = []
        for hit in results[0]:
            processed_results.append({
                "id": hit["id"],
                "pk": hit["pk"],
                "title": hit["title"],
                "text_preview": hit["text"][:200] + "...",
                "category": hit["category"],
                "location": hit["location"],
                "environment": hit["environment"],
                "distance": hit["distance"]
            })
        return processed_results

    except MilvusException as e:
        print(f"混合搜索时发生 Milvus 错误: {e}")
        return []
    except Exception as e:
        print(f"混合搜索时发生未知错误: {e}")
        return []

# 示例搜索调用
if collection.num_entities > 0:
    print("\n开始示例搜索...")
    search_results = hybrid_search("孙悟空的战斗技巧", category="神魔大战", limit=3)
    if search_results:
        for res in search_results:
            print(f"  - PK: {res['pk']}, Title: {res['title']}, Distance: {res['distance']:.4f}")
            print(f"    Category: {res['category']}, Location: {res['location']}")
            print(f"    Preview: {res['text_preview']}\n")
    
    search_results_filtered = hybrid_search("火焰山的战斗", environment="火山", limit=2)
    if search_results_filtered:
        for res in search_results_filtered:
            print(f"  - PK: {res['pk']}, Title: {res['title']}, Distance: {res['distance']:.4f}")
            print(f"    Category: {res['category']}, Location: {res['location']}, Environment: {res['environment']}")
            print(f"    Preview: {res['text_preview']}\n")
else:
    print("\n集合中没有实体，跳过示例搜索。")

print("\n脚本执行完毕。")


