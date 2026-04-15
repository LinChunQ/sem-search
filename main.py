import time
import numpy as np
from milvus_connection import get_collection
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from sentence_transformers import SentenceTransformer
import openai

# --- 1. 配置与模型加载 ---
# 加载本地 BGE-M3 模型 (使用 ModelScope 下载加速)
print("正在加载 BGE-M3 模型...")
model = SentenceTransformer('BAAI/bge-m3')

# LLM 配置 (以 OpenAI API 为例，也可以替换为本地 Qwen/Llama)
client = openai.OpenAI(api_key="你的API_KEY", base_url="你的代理地址")

# --- 2. Milvus 初始化 ---
collection_name = "complaint_deduplication"
dim = 1024  # BGE-M3 默认维度
collection = get_collection(collection_name=collection_name, dim=dim)


# --- 3. 核心功能函数 ---

def add_complaints(texts):
    """将投诉内容向量化并存入 Milvus"""
    embeddings = model.encode(texts)
    entities = [
        texts,
        embeddings.tolist()
    ]
    collection.insert(entities)
    collection.flush()
    print(f"成功导入 {len(texts)} 条初始投诉数据。")


def check_duplicate(new_complaint):
    """检索相似内容并调用 LLM 判定"""
    # 1. 向量检索
    query_vector = model.encode([new_complaint]).tolist()

    collection.load()
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=query_vector,
        anns_field="vector",
        param=search_params,
        limit=3,
        output_fields=["content"]
    )

    if not results[0]:
        return "未发现相似投诉"

    # 2. 遍历候选集进行 LLM 判定
    for hit in results[0]:
        score = hit.score
        ref_content = hit.entity.get('content')

        # 如果相似度极高（如 > 0.95），可直接判定为重复
        if score > 0.95:
            return f"【自动判定】高度重复 (Score: {score:.4f})"

        # 如果相似度处于模糊区间（如 0.7 - 0.95），调用 LLM 判定
        if score > 0.75:
            prompt = f"""你是一个投诉管理助手。请判断以下两段投诉内容是否描述的是同一次事件。
            注意：用户可能会在不同平台使用略有差异的描述，但如果核心时间、人物、诉求一致，则认为是同一个投诉。

            投诉文本 A: {new_complaint}
            投诉文本 B: {ref_content}

            请直接回答：'是同一个投诉' 或 '不是同一个投诉'，并简述理由。"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # 或你本地的 LLM
                messages=[{"role": "user", "content": prompt}]
            )
            return f"【LLM 判定结果】: {response.choices[0].message.content} (向量分: {score:.4f})"

    return "未发现语义重复投诉"


# --- 4. 运行验证 ---

# 模拟数据库中已有的投诉
existing_data = [
    "昨天在xx超市买的牛奶过期了，服务员态度还很差，要求退钱！",
    "2024年4月10日，我在某电商平台购买了一台洗地机，收到货后发现外壳破裂，联系客服多次不予处理。"
]
add_complaints(existing_data)

# 模拟新进入的、措辞稍有不同的投诉
test_complaint = "前几天在超市买到了变质的牛奶，找店员理论他们根本不理人，太气愤了，必须退款。"

print(f"\n待检测投诉: {test_complaint}")
decision = check_duplicate(test_complaint)
print(decision)
