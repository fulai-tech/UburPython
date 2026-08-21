# UburPython

**Somni / 功能手板音频检索服务**：三维度检索为核心；HTTP 保留；同进程再对外暴露功能手板与量产两套 gRPC。

- **核心**：三维度音频检索（ES 召回 + 标签词典向量 + 精排）
- **功能手板**：`MONGO_*` + `ES_NODE` + `REDIS_URL`；HTTP `:8080` + gRPC `:50051`
- **量产**：`SOMNI_MONGO_*` + `SOMNI_ES_*` + `SOMNI_REDIS_URL`；gRPC `:50052`
- **写路径**：直连 Mongo，再同步本侧 ES（不再调用 BioNode）
- **接口文档**：`docs/功能手板接口文档.md`、`docs/量产接口文档.md`

## 架构

```text
算法端 / 调用方
    │
    ├── HTTP :8080 ──► app/api/audio ──► server/handboard（Fullive）
    ├── gRPC :50051 ─► uburnode.v1（手板 audio/quiz）
    └── gRPC :50052 ─► uburnode.somni.v1（量产 ListTags/ListAudios/Search/GetAnswer）
```

## 目录结构

```text
UburPython/
├── app/
│   ├── main.py                 # FastAPI + lifespan（双 gRPC）
│   ├── api/audio.py            # HTTP /api/audio
│   ├── server/
│   │   ├── bootstrap.py        # 启停手板/量产 gRPC
│   │   ├── handboard/          # 功能手板 audio|quiz
│   │   └── somni/              # 量产 audio|quiz
│   ├── uburnode_grpc/grpc_gen/ # proto 生成 stub
│   ├── core/                   # 配置、日志、bson 工具
│   ├── schemas/
│   ├── services/retrieval.py   # 检索流水线
│   ├── es/                     # ES 读/写同步
│   ├── embedding/
│   ├── cache/
│   └── middleware/
├── scripts/
│   ├── sync_es_from_comm.py    # Mongo → ES 差异同步
│   └── gen_uburnode_proto.sh   # 生成对外 gRPC stub
├── proto/
│   ├── uburnode.proto
│   └── uburnode_somni.proto
├── tests/
├── pyproject.toml
└── .env.example
```

## 快速开始

```bash
# 1. 安装依赖（推荐 uv）
uv sync --extra dev

# 2. 生成对外 gRPC stub
chmod +x scripts/gen_uburnode_proto.sh
./scripts/gen_uburnode_proto.sh

# 3. 本地 Elasticsearch
docker compose -f docker-compose.es.yml up -d
curl -s http://localhost:9200

# 4. 配置环境变量
cp .env.example .env
# 编辑 ES_NODE、MONGO_URI、SOMNI_MONGO_URI、SOMNI_ES_NODE、EMBEDDING_* 等

# 5. 导出 ONNX 模型（若 models/ 目录尚无模型）
# 见 scripts/export_onnx_model.py

# 6. Mongo → ES 同步（手板库）
uv run python scripts/sync_es_from_comm.py --dry-run
uv run python scripts/sync_es_from_comm.py

# 7. 启动服务
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

开发模式（`APP_DEBUG=true`）跳过 Embedding 模型加载，便于本地调试 HTTP 路由。

## HTTP 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/audio` | POST | 创建音频（Mongo Somni；仅 `audio_name` 必填） |
| `/api/audio/{id}` | PUT | 更新音频（字段全选填，partial update） |
| `/api/audio/{id}` | DELETE | 删除音频（comm + ES） |
| `/api/audio/search` | POST | 三维度检索 |

OpenAPI 文档：启动后访问 `http://localhost:8080/docs`。

### 创建接口

**请求** `POST /api/audio`（仅 `audio_name` 必填，其余选填）：

```json
{
  "audio_name": "阴雨天城市公寓的雷雨氛围感音效",
  "audio_url": "https://cdn.fulai.tech/somni/audio/demo.mp3",
  "operation_type": 0,
  "created_by": "qwen3.5-omni-plus",
  "description": "...",
  "sleep_stage_tags": [{"tag_id": "...", "code": "unwind", "name": "放松"}],
  "content_form_tags": [],
  "mechanism_tags": [],
  "audio_engineering_tags": [],
  "medical_risk_tags": [],
  "evidence_level_tags": [{"tag_id": "...", "code": "B", "name": "中等证据"}],
  "embedding": []
}
```

有 `audio_url` 时会同步写入 ES（生成 `description_text` / `description_vector`）。

### 更新接口

**请求** `PUT /api/audio/{id}`：同一套字段，**全部可选**，只更新请求中出现的字段。

### 检索接口

**请求** `POST /api/audio/search`（单次检索）：

```json
{
  "sleep_stage_tags": ["放松"],
  "content_tags": ["慢钢琴"],
  "disliked_tags": [],
  "top_k": 10
}
```

**响应** `data.materials` 为 `somni_audio_materials` 索引文档列表：

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
| 2 | `content_tags` 与内容/机制/工程标签精确或向量模糊命中；白/粉/棕噪音互斥，仅允许精确命中 |
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
| `SIM_THRESHOLD` | `0.7` | 内容形态向量模糊命中阈值 |
| `STRONG_DISLIKE_SIM_THRESHOLD` | `0.85` | 多路文本检索厌恶硬剔除阈值 |
| `EMBEDDING_ONNX_DIR` | `models/onnx/bge-small-zh-v1.5` | ONNX 模型目录 |
| `EMBEDDING_ONNX_POOL_SIZE` | `4` | ONNX 会话池大小（并发推理路上限） |
| `EMBEDDING_ONNX_INTRA_OP_THREADS` | `1` | 单 session ORT 线程数 |

完整列表见 [`.env.example`](.env.example)。

## 日志

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LOG_DIR` | `logs` | 日志目录（部署时挂载到宿主机项目 `logs/`） |
| `LOG_RETENTION` | `7 days` | 历史按日日志保留时长 |

文件命名为 `YYYY-MM-DD_ubur_log`（如 `2026-07-15_ubur_log`），每天一个明文文件，零点滚动。  
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
