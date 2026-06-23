# UburPython

BioNode 体系中的 **Somni 音频检索服务**：以三维度检索为核心，从 MongoDB 同步 Somni 原料与标签词典至 Elasticsearch，对外提供 HTTP 检索 API。

- **核心**：三维度音频检索（`somni_audio_materials` ES 召回 + 标签词典向量 + 四步精排流水线）
- **数据源**：MongoDB `Fullive` 库（`somni_audio_materials`、`somni_audio_tag_dictionary`）
- **索引**：Elasticsearch `somni_audio_materials`、`somni_audio_tag_dictionary`（字段含义见 mapping `meta.description`）
- **写路径（遗留）**：HTTP CUD 仍经 comm-service gRPC；Somni 索引由同步脚本维护

## 架构

```text
算法端 / 调用方
    │
    ▼
对外 HTTP (FastAPI + Pydantic)
    │
    ├──读（检索）──► somni_audio_materials (ES)
    │                  + somni_audio_tag_dictionary (ES 向量)
    │                  + 进程内 Embedding (bge-small-zh-v1.5)
    │
    ├──写（CUD，遗留）──► comm-service (gRPC) ──► MongoDB（旧 comm 表）
    │
    └──同步（定时/手动）──► MongoDB Somni 集合 ──► Elasticsearch
```

## 数据流

| 环节 | 来源 | 目标 | 模块 |
|------|------|------|------|
| Mongo → ES 同步 | `somni_audio_materials`、`somni_audio_tag_dictionary` | 同名 ES 索引 | `scripts/sync_es_from_comm.py` |
| 标签向量 | 词典 `name` / `name_en` | `name_vector` / `name_en_vector` | 同步脚本 + `app/embedding/` |
| HTTP 检索 | ES 原料文档 | `data.materials[]` 原样返回 | `app/services/retrieval.py` |
| HTTP CUD（遗留） | 六维标签入参 | comm 扁平 tags | `app/services/audio.py` |

字段命名全链路 **snake_case**。Somni 表结构详见仓库内 `音频表结构.md`。

## 目录结构

```text
UburPython/
├── app/
│   ├── main.py                 # FastAPI 入口 + lifespan
│   ├── core/                   # 配置、日志、标签转换
│   ├── api/audio.py            # 4 个 HTTP 端点
│   ├── schemas/audio.py        # Pydantic 模型
│   ├── services/               # AudioService、RetrievalService
│   ├── es/
│   │   ├── search.py           # EsSearch 读路径
│   │   ├── sync.py             # EsSync 写路径（CUD 跳过 upsert）
│   │   └── index_mappings.py   # ES 索引 mapping + 字段注释
│   ├── embedding/encoder.py    # bge-small-zh-v1.5 向量编码
│   └── bionode_grpc_clients/   # comm-service gRPC 客户端
├── scripts/
│   ├── sync_es_from_comm.py    # Mongo → ES 差异同步
│   └── gen_proto.sh            # 生成 gRPC stub
├── proto/                      # bionode_comm.proto
├── tests/
├── pyproject.toml
└── .env.example
```

## 快速开始

```bash
# 1. 安装依赖（推荐 uv）
uv sync --extra dev

# 2. 生成 comm gRPC stub（CUD 接口需要）
chmod +x scripts/gen_proto.sh
./scripts/gen_proto.sh

# 3. 本地 Elasticsearch
docker compose -f docker-compose.es.yml up -d
curl -s http://localhost:9200

# 4. 配置环境变量
cp .env.example .env
# 编辑 ES_NODE、MONGO_URI、EMBEDDING_ONNX_DIR、COMM_GRPC_* 等

# 5. 导出 ONNX 模型（若 models/ 目录尚无模型）
# 见 scripts/export_onnx_model.py

# 6. Mongo → ES 全量同步
uv run python scripts/sync_es_from_comm.py --dry-run
uv run python scripts/sync_es_from_comm.py

# 7. 启动服务
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

开发模式（`APP_DEBUG=true`）跳过 Embedding 模型加载，便于本地调试 HTTP 路由。

## HTTP 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/audio` | POST | 创建音频（comm gRPC，遗留） |
| `/api/audio/{id}` | PUT | 更新音频（comm gRPC，遗留） |
| `/api/audio/{id}` | DELETE | 删除音频 |
| `/api/audio/search` | POST | 三维度检索 |

OpenAPI 文档：启动后访问 `http://localhost:8080/docs`。

### 检索接口

**请求** `POST /api/audio/search`：

```json
{
  "sleep_stage_tags": ["放松"],
  "content_tags": ["慢钢琴"],
  "disliked_tags": [],
  "top_k": 10
}
```

**响应** `data.materials` 为命中条目的 `somni_audio_materials` 索引文档（含 `id`，字段与 ES/Mongo 一致，暂不做裁剪）：

```json
{
  "code": 200,
  "msg": "检索成功",
  "data": {
    "materials": [
      {
        "id": "6a33a7928030d4cf420efeb6",
        "audio_name": "专属冥想南极 助眠解压舒缓情绪",
        "description": "...",
        "status": true,
        "audio_url": "https://cdn.fulai.tech/comm/audio/xxx.mp3",
        "operation_type": 0,
        "created_by": "qwen3.5-omni-plus",
        "updated_by": "qwen3.5-omni-plus",
        "sleep_stage_tags": [{ "tag_id": "...", "code": "unwind", "name": "放松" }],
        "content_form_tags": [],
        "mechanism_tags": [],
        "audio_engineering_tags": [],
        "medical_risk_tags": [],
        "evidence_level_tags": [{ "tag_id": "...", "code": "B", "name": "中等证据" }],
        "created_at": "2026-06-18T00:00:00.000Z",
        "updated_at": "2026-06-18T00:00:00.000Z"
      }
    ]
  },
  "timestamp": "..."
}
```

## 检索流水线

```text
睡眠阶段精确过滤 → 内容形态准入 → 厌恶剔除 + 粗排 → 精排
```

| 步骤 | 说明 |
|------|------|
| 1 | `sleep_stage_tags.name` nested 精确匹配（可配置跳过） |
| 2 | `content_tags` 与内容/机制/工程标签精确或向量模糊命中 |
| 3 | `disliked_tags` 向量相似则剔除 |
| 4 | 按 `match_count` 降序，`top_k` 截断 |

## Mongo → ES 同步

```bash
uv run python scripts/sync_es_from_comm.py          # 正式同步
uv run python scripts/sync_es_from_comm.py --dry-run # 仅比对统计
```

- 先同步 `somni_audio_tag_dictionary`（写入 `name_vector`、`name_en_vector`）
- 再同步 `somni_audio_materials`（1:1 镜像 Mongo 文档）
- 启动时删除旧索引 `audio_materials`、`tag_vectors`

服务内按 `SYNC_INTERVAL_DAYS` 定时执行（需配置 `MONGO_URI`）。

## 环境变量（节选）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ES_NODE` | `http://localhost:9200` | Elasticsearch 地址 |
| `ES_AUDIO_INDEX` | `somni_audio_materials` | 原料索引名 |
| `ES_TAG_VECTORS_INDEX` | `somni_audio_tag_dictionary` | 标签词典索引名 |
| `MONGO_URI` | — | MongoDB 连接串（同步必填） |
| `MONGO_DB` | `Fullive` | 数据库名 |
| `SIM_THRESHOLD` | `0.7` | 向量模糊命中阈值 |
| `EMBEDDING_ONNX_DIR` | `models/onnx/bge-small-zh-v1.5` | ONNX 模型目录 |

完整列表见 [`.env.example`](.env.example)。

## 日志

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LOG_DIR` | `logs` | 日志目录 |
| `LOG_FILE_NAME` | `uburnode.log` | 日志文件名 |
| `LOG_ROTATION` | `10 MB` | 单文件滚动大小 |
| `LOG_RETENTION` | `7 days` | 历史日志保留 |

响应头回传 `X-Request-Id` 便于链路追踪。

## Docker 部署

```bash
cd /opt/uburpython && docker compose up -d --build
```

生产访问：`http://<服务器IP>:8001/docs`（nginx 映射宿主机 8001 → 容器 80）。

## 测试

```bash
uv run pytest
```
