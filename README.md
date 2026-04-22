# Claim Duplicate Test

一个基于 `Flask + Milvus + BGE-M3` 的语义检索示例项目，用于把文本写入向量库，并按相似度返回 `TopN` 结果。项目同时支持两种检索模式：

- `dense`：只使用稠密向量进行检索。
- `hybrid`：使用 `BGE-M3` 的稠密向量和稀疏向量做混合检索。

当前仓库提供了一个简单 Web 页面，支持：

- 单条文本写入 Milvus。
- 批量导入 JSON 文本数据。
- 输入查询文本，返回最相似的 `TopN` 结果。
- 通过 `/health` 查看服务基础状态。

## 项目结构

```text
.
├─ main.py                  # Flask 入口
├─ milvus_connection.py     # Milvus 连接、建库建表、索引初始化
├─ sem_search_service.py    # Embedding 编码、写入、检索逻辑
├─ vector_service.py        # 兼容别名
├─ templates/
│  └─ index.html            # 页面模板
├─ models/                  # 本地模型目录（默认不提交）
├─ test_data_claims_100.json
├─ test_data_long_20.json
└─ requirements.txt
```

## 环境要求

- Python 3.10 及以上
- 可访问的 Milvus 实例
- 已安装 `pip`
- 如需本地运行 `BGE-M3`，建议预留足够磁盘空间

## 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果 `torch` 安装在你的环境里失败，优先按 PyTorch 官方方式安装与本机 CUDA/CPU 匹配的版本，然后再执行：

```powershell
pip install -r requirements.txt
```

## 配置说明

项目通过根目录下的 `.env` 读取配置。下面是一个可参考的配置模板：

```env
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=
MILVUS_PASSWORD=
MILVUS_URI=
MILVUS_TOKEN=
MILVUS_SECURE=false
MILVUS_DB_NAME=default
MILVUS_COLLECTION_NAME=claim_deduplication
MILVUS_RECREATE_COLLECTION=false

EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_MODEL_PATH=

HF_ENDPOINT=https://hf-mirror.com
HF_TOKEN=
HF_HUB_DISABLE_SYMLINKS_WARNING=1

SEARCH_MODE=dense
BGE_M3_USE_FP16=false
HYBRID_DENSE_WEIGHT=0.7
HYBRID_SPARSE_WEIGHT=0.3
HYBRID_CANDIDATE_LIMIT=20
```

### 关键配置项

- `SEARCH_MODE`
  - `dense`：仅执行稠密向量检索。
  - `hybrid`：启用 `BGE-M3` 混合检索，需要 `FlagEmbedding`，并要求 Milvus 支持稀疏向量索引。
- `EMBEDDING_MODEL_PATH`
  - 指定本地模型目录。
  - 如果不填，程序会优先查找默认目录 `models/bge-m3`。
  - 如果默认目录也不存在，则回退到远程模型名 `BAAI/bge-m3`。
- `MILVUS_RECREATE_COLLECTION`
  - 设为 `true` 时，启动时会重建集合。
  - 当你从旧的 dense-only schema 切到 `hybrid` 模式时，这个配置很有用。
- `HYBRID_DENSE_WEIGHT` / `HYBRID_SPARSE_WEIGHT`
  - 控制混合检索重排序时的稠密/稀疏权重。

## BGE-M3 本地模型说明

这个项目使用的 `BGE-M3` 本地模型文件体积较大，`models/bge-m3` 目录不适合直接上传到 GitHub，因此仓库默认忽略 `models/` 目录。

这意味着：

- GitHub 仓库里通常不会包含完整的本地模型文件。
- 克隆仓库后，需要你自行下载模型到本地。
- 推荐把模型放在 `models/bge-m3`，或者通过 `.env` 的 `EMBEDDING_MODEL_PATH` 指向你的本地目录。

一个常见的下载方式如下：

```powershell
$env:HF_ENDPOINT='https://hf-mirror.com'
.\.venv\Scripts\hf download BAAI/bge-m3 --local-dir .\models\bge-m3
```

如果本地已经有模型，也可以直接在 `.env` 里配置：

```env
EMBEDDING_MODEL_PATH=D:\path\to\bge-m3
```

## 运行项目

```powershell
.\.venv\Scripts\python main.py
```

默认启动地址：

```text
http://127.0.0.1:5000/
```

## 页面功能

### 1. 写入向量库

支持两种方式：

- 在文本框中输入一段完整文本后直接提交。
- 上传 JSON 文件做批量导入。

批量导入 JSON 格式如下：

```json
[
  { "data": "第一条文本" },
  { "data": "第二条文本" }
]
```

每个对象都需要包含 `data` 字段，且值必须是非空字符串。

### 2. 相似度检索

在页面右侧输入查询文本并设置 `TopN`，系统会：

1. 使用当前 Embedding 模型将查询文本转成向量。
2. 在 Milvus 中执行向量检索。
3. 返回最相似的文本内容及相似度分数。

`TopN` 当前限制为 `1` 到 `20`。

## 接口说明

- `GET /`
  - 返回检索页面。
- `POST /add`
  - 写入单条或批量文本到 Milvus。
- `POST /search`
  - 执行相似度检索。
- `GET /health`
  - 返回服务状态、集合名、模型名和检索模式。

## 检索模式说明

### Dense 模式

- 使用 `sentence-transformers` 加载模型。
- 仅写入和检索 `vector` 字段。
- 依赖更少，适合快速验证流程。

### Hybrid 模式

- 使用 `FlagEmbedding` 的 `BGEM3FlagModel`。
- 同时写入 `vector` 和 `sparse_vector`。
- 检索时结合稠密向量和稀疏向量结果做加权重排。
- Milvus 版本需要支持 `SPARSE_FLOAT_VECTOR` 和 `SPARSE_INVERTED_INDEX`。

## 已提供测试数据

- `test_data_claims_100.json`
- `test_data_long_20.json`

可直接用于页面批量导入，验证写入与检索流程。

## 常见问题

### 1. 启动时报找不到 `.env`

项目启动时会直接读取根目录 `.env`。如果文件不存在，会抛出配置文件缺失异常。请先创建 `.env` 再启动。

### 2. 切换到 hybrid 后报集合 schema 不兼容

这是因为旧集合可能只有稠密向量字段。可以：

- 设置 `MILVUS_RECREATE_COLLECTION=true` 后重启服务。
- 或者换一个新的 `MILVUS_COLLECTION_NAME`。

### 3. 模型下载失败

如果默认 Hugging Face 源访问较慢，可以在 `.env` 里配置：

```env
HF_ENDPOINT=https://hf-mirror.com
```

## 说明

本项目当前更适合作为本地验证和检索流程演示，不建议把大模型文件直接纳入 Git 仓库。仓库应提交代码、配置说明和依赖清单，而模型文件建议通过本地下载或私有制品存储方式管理。
