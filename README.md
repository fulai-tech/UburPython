# UburPython

**Somni / 功能手板音频检索服务**：三维度检索为核心；HTTP 保留；同进程再对外暴露功能手板与量产两套 gRPC。

- **核心**：三维度音频检索（ES 召回 + 标签词典向量 + 精排）
- **功能手板**：`MONGO_*` + `ES_NODE` + `REDIS_URL`；HTTP `:8080` + gRPC `:50065`
- **量产**：`SOMNI_MONGO_*` + `SOMNI_ES_*` + `SOMNI_REDIS_URL`（独立 Redis 实例）；gRPC `:50064`
- **写路径**：直连 Mongo，再同步本侧 ES（不再调用 BioNode）
- **接口文档**：`docs/功能手板接口文档.md`、`docs/量产接口文档.md`

### Profile and plan hashes as metadata (local, not deployed)

All `GetProfileInputs` purposes and inline profiles in `GetPlanInputs` skip stored
body and transport-envelope hash comparisons. Original bodies and hash values are
returned without database writes; envelope hashes are still generated. The section
reason is `long_term_hash_verification_skipped`. Owner/schema/version-link/status,
object shape, finite JSON, visibility, source and snapshot-ambiguity checks remain.
The user explicitly retained existing UID/schema/hash database association queries.
Missing linked records and malformed bodies do not become empty successful profiles.
This does not repair missing preferences, obsolete dates or duplicate active snapshots.
DPH must also use the metadata-only policy; updating only DPH cannot change this service.

Regression: `python -m pytest tests/test_somni_facts_service.py -q`.

### Current-version profile and plan inputs

Profile/plan generation no longer accepts or emits `data_cutoff_at`,
`data_cutoff_at_utc` or `cutoff_policy_version`. Old JSON/DB metadata is ignored;
retired protobuf field numbers and names are reserved. Source receipt/update times
are provenance, without an implicit read-start visibility cap. Business-date windows,
active status, revision/association rules, ownership, finite values, quality checks and
query budgets remain. Historical business dates use currently available versions;
this is a best-effort read, not an archival or transactional snapshot. No DB migration
or repair is performed. Same-day unchanged profiles can be reused independent of
read time; changed business evidence can produce a new revision.

The policy is scoped to each async profile/plan request and resets on errors.
Reminder and evidence-pagination read boundaries retain their independent behavior.
Observation validity uses the target business date at the supplied fixed UTC offset.
Both middleware and DeepHealth must be updated together; no deployment is included.

Regression: `uv run --no-sync pytest tests/test_cutoff_removal.py tests/test_grpc_generation_data.py -q`.

### Legacy `somni_records` profile-facts fallback (local, not deployed)

When report or normalized daily metrics do not cover a metric/date, `build_facts`
can project validated `somni_records.raw_data` fields into the existing typed
daily series. The projection reuses the current report definitions: total sleep
is the floored sum of REM/light/deep stage minutes over bed-to-final-wake time;
SOL uses `sleep_latency`; WASO uses `wake_time` to `wake_up_time`; sleep
efficiency uses `sleep_efficiency` percent. It requires finite in-range values,
an aware ordered sleep interval, a wake time whose local date
matches `record_date`, and source refs. The row's `timezone_offset` is used when
present; otherwise the request's fixed offset is used and historical travel/DST
cannot be reconstructed. If a quality object explicitly marks the record
invalid/unverified, it is rejected; absence of a legacy quality object is not
proof of device quality. Values only certify the field projection checks, not
clinical accuracy. Missing fields stay missing, never zero.

Per metric and local wake date, a validated normalized wearable value wins,
then a validated daily-report value, then a validated legacy-record projection.
Only one source contributes a value for a given date; conflicting duplicate
records remain missing/partial. The internal `facts.series` shape is unchanged;
top-level source/method metadata is `mixed` when daily rows use more than one
source, while each daily value retains its own source and refs. This does not
generate reports or aggregate raw radar/HR/RR/IBI signals. The facts section
continues to report unrelated pending aggregation/visibility limitations.

### v6 presleep evidence (local, not deployed)

Internal Somni Proto adds `InputFacts.metric_windows=7` and optional
`InputNumericStats.std_dev=6`; absent fields remain readable, not zero evidence.
`InputMetricWindow` carries metric/unit/source/method/definition, date/episode,
UTC bounds, optional value/coverage/count, quality/measurement condition,
HRV method and source refs.

`normalize_metric_windows` validates existing aggregate `metric_windows` from
records/wearable_daily (or records raw_data). Full duration coverage, provenance,
presleep awake-rest condition and RMSSD/ms for HRV are required; conflicting
duplicates are excluded; profile/plan reads do not filter by read time. Raw HR/RR/IBI aggregation and
main-sleep inference remain pending. Missing windows adds
`presleep_metric_windows_unavailable`, retaining original partial reasons.
Generic numeric aggregation does not populate std_dev; DeepHealth computes
reviewed sample SD from seven preceding nights. No database writes, migration
or seeding. Stable questionnaire/DQ codes and device-quality/window producers
remain dependencies. Producer/consumer rollout requires separate authorization.

## 架构

```text
算法端 / 调用方
    │
    ├── HTTP :8080 ──► app/api/audio ──► server/handboard（Fullive）
    ├── gRPC :50065 ─► uburnode.v1（手板 audio/quiz）
    └── gRPC :50064 ─► uburnode.somni.v1（量产 ListTags/ListAudios/Search/GetAnswer）
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
│   ├── sync_es_from_comm.py    # 手板 Mongo → ES 全量同步
│   ├── sync_es_from_somni.py   # 量产 Mongo → ES 全量同步
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

# 6. Mongo → ES 同步
# 手板库
uv run python scripts/sync_es_from_comm.py --dry-run
uv run python scripts/sync_es_from_comm.py
# 量产库
uv run python scripts/sync_es_from_somni.py --dry-run
uv run python scripts/sync_es_from_somni.py

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
# 手板（MONGO_URI / ES_NODE）
uv run python scripts/sync_es_from_comm.py          # 正式同步
uv run python scripts/sync_es_from_comm.py --dry-run # 仅比对统计

# 量产（SOMNI_MONGO_URI / SOMNI_ES_NODE）
uv run python scripts/sync_es_from_somni.py
uv run python scripts/sync_es_from_somni.py --dry-run
```

- 先同步 `somni_audio_tag_dictionary`（写入 `name_vector`、`name_en_vector`）
- 再同步 `somni_audio_materials`（1:1 镜像 Mongo 文档）
- 启动时删除旧索引 `audio_materials`、`tag_vectors`

手板服务内按 `SYNC_INTERVAL_DAYS` 定时执行（需配置 `MONGO_URI`）。量产同步目前仅 CLI。

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


## GenerationData Internal Access

GenerationData RPCs are internal interfaces for DeepHealth. They do not authenticate
callers; network access to the Somni gRPC port must be restricted to trusted services.
Do not expose this port publicly. The caller supplies the UID. UID validation,
record ownership, profile/hash links, plan versions, cursor signatures and query
budgets remain enforced. Account eligibility and user consent are enforced by the
caller; GenerationData does not require a `somni_users` record or `privacy_switches`.

Saved-profile bodies must remain finite JSON objects, but hashes are output metadata,
not validation gates in either profile or plan generation. DPH retains existing hash
generation algorithms and returns stored hashes without comparing body/envelope
digests. The middleware generates normalized envelope metadata without digest rejection.
Hash mismatches no longer produce partial/long_term_hash_mismatch; missing bodies,
invalid dates, ownership failures and duplicate snapshots remain separate errors.
Existing UID/schema/hash association queries remain; no historical-data repair or
checksum rewrite is performed. Hash fields are not authentication or proof of integrity.

`GetProfileInputs` and `GetPlanInputs` use the current-version generation policy above.
The envelope's `read_started_at_utc` and `generated_at_utc` retain actual audit times.

Database writers may preserve receipt times and immutable versions; this change does not create an archive.

Mongo compatibility: when no `received_at` exists, the persisted `update_time`
(otherwise `create_time`) is a provenance timestamp, not proof
of network receipt. Later updates are eligible in current-version profile/plan reads. Completed
answer ObjectIds map to `answer_id`; completion time uses the persisted version
time when absent. Missing optional survey versions are omitted without inventing
values or rejecting the completed answer. `InputAnswer` adds stored `title` (field 8),
`input_type` (field 9) and `extra_input` (field 10); fields 1-7 are unchanged, so older
clients can ignore the additional fields. No database migration is required.
Current `profile_snapshot_id` and legacy `snapshot_id` are supported,
but conflicts/ambiguous exact matches are rejected. Schema0.0.1 keeps its original
`sha256:` long-body digest; a digest mismatch remains an integrity failure.

## 测试

```bash
uv run pytest
```

### 偏好证据聚合接入状态（2026-10-16）

`app/server/somni/facts/preference.py` 和 `FactsStore.load_preference_sources` 已提供独立的只读聚合：
按 UID、本地14晚业务窗口查询 UTC 数据，保留预算；重复包去重、近时旋钮首尾合并，
精确关联方案版本、设备和完整会话，输出快照人工差异及暴露事实。
偏好确认阈值归 DeepHealth；本仓不决定是否自动调整。

已接入 gRPC。用户批准新增可选 `InputPreferenceEvidenceSection`，在 `ProfileInputsData` 使用字段8、
`PlanInputsData` 使用字段6，两仓proto和生成文件已同步；旧端缺失证据时不能推断确认。
真实快照字段、节点完成覆盖及硬件保持时长证明尚待数据核验，缺失时保守降级。
本仓只返回事实证据，偏好判定与画像/方案候选由DPH完成；没有创建调度器、自动修改已保存方案或数据库迁移。

### 生产结构偏好适配与验证数据

偏好聚合支持快照`date/create_time/status=0/edit_source=app`，会话通过唯一归属的`plan_id + plan_version`关联。完整暴露必须同时验证completed、非预览、设备、时区、方案时间及节点总数；停止/失败/缺少时区不计。已采纳App修改仅比较连续唯一版本；排队/丢弃/拒绝快照不作采纳证据。硬件成功记录可归并为观察，但缺少持续生效证明仍不自动确认。

daily兼容没有generation链接的旧当前方案，通过用户/日期/版本/快照正文一致核验返回现有目标协议；weekly/realtime的独立读取语义不扩展。当前方案内可选null占位不构成正文差异，真正数值/节点差异仍拒绝。候选保存需版本和正文hash CAS，并写`trigger_reason=daily_preference`防重复递减。

五人包由DeepHealth的`scripts/somni_preference_seed.py`生成；本仓`scripts/somni_preference_fixture_import.py`默认只检查包、不连接数据库，只有`--apply`才会导入，连接通过已有`SOMNI_MONGODB_URI`进程环境读取。包采用Extended JSON BSON UTC日期、确定性隔离UID和_id，覆盖问卷、长短画像、7天方案、5晚会话、App修订及硬件重传。测试日期必须与daily请求当日一致（建议09:00），过期包需重新生成；不创建登录账户或真实设备绑定。
