# UburNode

BioNode 体系中的**中间层服务**：以**音频检索**为核心能力，同时承担算法端与底层存储之间的**数据结构转换**。

- **核心**：三维度音频检索（ES 召回 + 进程内 Embedding + 四步精排流水线）
- **中间层**：统一对外 HTTP/Pydantic 契约，对内经 gRPC 访问 comm-service；在 HTTP、Mongo（扁平标签）、ES（六维结构 + 向量）之间做形态互转，避免算法端直连 Mongo 或各自维护多套字段约定

## 架构

```text
算法端 / 调用方
    │
    ▼
对外 HTTP (FastAPI + Pydantic)     ← 中间层：契约统一 + 数据结构转换
    │
    ├──读──► Elasticsearch + Embedding + RetrievalService
    │
    └──写──► comm-service (gRPC) ──► MongoDB（真值库，扁平 tags）
                  └── EsSync ──► Elasticsearch（索引副本，六维 tags + vector_id）
```

## 中间层数据转换

UburNode 不只做检索，还在各存储边界维持**单一对外契约**并完成形态映射：

| 边界 | 入站形态 | 出站形态 | 负责模块 |
|------|----------|----------|----------|
| HTTP 写（CUD） | 六维标签对象 `AudioTagsInput` | comm/Mongo 扁平 `string[]`（带维度前缀） | `app/schemas/audio.py`、`app/core/tags.py` |
| HTTP 写响应 | comm `AudioMaterialInfo`（gRPC） | HTTP `AudioMaterialData` | `app/schemas/audio.py` |
| ES 同步 | 扁平 tags + 业务字段 | ES 六维 `tags` + `tag_vectors` embedding | `app/es/sync.py` |
| HTTP 读（检索） | ES 六维 `TagItem`（含 `vector_id`） | 出参六维 label 字符串 | `app/schemas/audio.py`、`app/services/retrieval.py` |

字段命名全链路 **snake_case**；对外以 Pydantic + OpenAPI 为唯一 HTTP 契约，对内 comm 调用走同源 `bionode_comm.proto`。

## 目录结构

```text
UburNode/
├── app/
│   ├── main.py              # FastAPI 入口 + lifespan
│   ├── core/                # 配置、日志
│   ├── api/audio.py         # 4 个 HTTP 端点
│   ├── schemas/audio.py     # Pydantic 模型
│   ├── services/            # AudioService、RetrievalService
│   ├── es/                  # EsSearch、EsSync
│   ├── embedding/encoder.py # bge-small-zh-v1.5 向量编码
│   ├── bionode_grpc_clients/  # BioNode 外部微服务 gRPC 客户端
│   │   └── comm/            # comm-service（client.py + grpc_gen/）
├── proto/                   # bionode_comm.proto（唯一真源）
├── scripts/gen_proto.sh     # 生成 gRPC stub
├── tests/
├── .cursor/skills/          # 项目 Skill
├── pyproject.toml
└── .env.example
```

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. 生成 comm gRPC stub
chmod +x scripts/gen_proto.sh
./scripts/gen_proto.sh

# 3. 本地 Elasticsearch（向量索引，需先就绪）
# 未安装 Docker 时：brew install --cask docker-desktop，打开 Docker Desktop 等待就绪
docker compose -f docker-compose.es.yml up -d
curl -s http://localhost:9200   # 应返回 cluster 信息
# .env 默认 ES_NODE=http://localhost:9200；索引由应用启动时 ensure_indices 自动创建

# 4. 配置环境变量
cp .env.example .env
# 编辑 ES_NODE、COMM_GRPC_HOST 等

# 5. 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

开发模式（`APP_DEBUG=true`）跳过 Embedding 模型加载，便于本地调试 HTTP 路由。

## HTTP 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/audio` | POST | 创建音频并同步 ES |
| `/api/audio/{id}` | PUT | 更新音频并同步 ES |
| `/api/audio/{id}` | DELETE | 删除音频并同步 ES |
| `/api/audio/search` | POST | 三维度检索 |

OpenAPI 文档：启动后访问 `http://localhost:8080/docs`。

### comm-service gRPC 连通探测

```bash
source .venv/bin/activate   # 或: .venv/bin/python scripts/test_grpc_connect.py
python scripts/test_grpc_connect.py
# 或集成测试（需可达的 COMM_GRPC_HOST）
COMM_GRPC_INTEGRATION=1 pytest tests/test_comm_grpc.py -v
```

## 检索流水线

```text
睡眠阶段精确过滤 → 内容形态准入 → 厌恶剔除 + 粗排 → 精排
```

## Proto 变更

`comm-service` 修改 `proto/bionode_comm.proto` 后须重新生成 stub：

```bash
./scripts/gen_proto.sh
```

## 日志

每次 HTTP 请求自动写入日志文件（`RequestLogMiddleware`），包含 method、path、status、耗时、`request_id`。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LOG_DIR` | `logs` | 日志目录 |
| `LOG_FILE_NAME` | `uburnode.log` | 日志文件名 |
| `LOG_ROTATION` | `10 MB` | 单文件滚动大小 |
| `LOG_RETENTION` | `7 days` | 历史日志保留 |

日志同时输出到控制台和 `logs/uburnode.log`。响应头会回传 `X-Request-Id` 便于链路追踪。

## Docker 部署（服务器）

```bash
# 1. 本机一键写入 GitHub Secrets（需浏览器登录 gh 一次）
chmod +x scripts/setup_github_secrets.sh
./scripts/setup_github_secrets.sh

# 2. 服务器一次性准备
#    - 将 setup 脚本输出的公钥写入 ~/.ssh/authorized_keys
#    - mkdir -p /opt/uburnode && 复制 .env.example 为 /opt/uburnode/.env 并填写 COMM_GRPC_* 等
#    - 无需手动装 Docker：首次 Deploy 会幂等执行 scripts/server_bootstrap.sh

# 3. GitHub → Actions → Deploy UburNode → Run workflow
#    后续每次 Deploy 仅 pull 镜像 + compose up，已装 Docker 则跳过安装
```

生产访问：`http://<服务器IP>:8001/docs`（nginx 映射宿主机 8001 → 容器 80，避免与宝塔 80 冲突）。

## 测试

```bash
pytest
```
