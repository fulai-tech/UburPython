# 音频素材读取与启用内容标签契约

本次修复从远端最新 `origin/dev` `ef4416da5b51d73acf7b066b15c18e09ae8d4014` 开始。代码提交与部署验收分开；尚未声称开发中间件已升级。

## Enabled content dictionary alignment (2026-11-03)

`app/core/audio_tag_aliases.json` is the versioned export of
`somni_audio_tag_dictionary` with the exact filter
`{"type":"content_form","status":"启用"}` (currently 106 records). Do not replace
it with a smaller manually curated catalogue. Unknown, disabled, retired and
wrong-dimension machine content codes remain invalid; this repair does not turn
off request validation. Application sleep-stage values `unwind/soothe/guard/wake`
are validated separately. `voice` is an existing safety exclusion, not a content
tag offered as an intervention.

The previous snapshot is archived in `audio_tag_aliases_v5.json` for engineering
alias compatibility, not for admitting retired content codes. Positive preferences keep exact current code/name identities. Negative preferences
union the exact identity with reviewed whole-term groups. The shared
contract revision is 5, so retrieval caches cannot reuse the old identity set.
Vocal filtering uses valid enabled content tags and their ancestry; materials
without usable content evidence fail closed for a no-voice request; retired
engineering voice/voice_position values are no longer read. Fresh Mongo
material content references are checked again after ES retrieval/cache lookup.
Database status and stage checks remain in force. RPC/Proto fields are unchanged.

Before release, use the explicit Somni connection profile to check both the live
dictionary and the DPH mirror without writing files or databases:

```bash
PYTHONPATH=. python scripts/export_audio_tag_contract.py \
  --mirror-root /path/to/DeepHealth --check-live --env-file /path/to/operator.env
```

The check fails on live drift or mismatched assets/shared identity functions.
Updating assets requires an explicit `--refresh --write` and review of preference
rules and DPH frozen playback policy versions; the exporter never overwrites
historical assets. It never updates Mongo, embeddings, or ES. Runtime processes
must restart after an approved asset update because their dictionary is cached.

Restored examples include `mindfulness_meditation`, `deep_slow_river`,
`postponement`, `imagery_distraction`, `harp_music`, `singing_bowl`,
`wind_ambience`, and `wuxing_healing_music`. Dictionary validity does not prove
that an enabled stage-appropriate audio material exists for every tag; an empty
candidate result remains a legitimate result.

## 审查修复与只读对账（2026-11-03）

- 雨声组覆盖 steady_light_rain、steady_rain、heavy_rain、surface_rain、scene_rain、rainforest_light_rain。负向“鸟类叫声/Bird Calls/bird_calls”覆盖 bird_calls 与 dawn_chorus；正向正式标签仍精确匹配。只扩展经审核的完整词，不做任意子串、模糊或模型推断。
- 实时库106项；相对109项移出了 storm_ambience、shore_water、sleep_education。97条启用素材中，原13个退役码引用为0、storm_ambience引用为0，但2条素材仍带 shore_water。它们会被严格拒绝；应先修复数据引用，再重建索引/缓存。此代码不删除标签、不改素材、不补造对应关系。
- `_verify_prescription_materials`拒绝非法内容引用时记录 warning：仅素材ID、稳定原因、内容code；不记录用户偏好、标题、URL或完整素材。不能为了召回而静默丢弃非法标签后放行整条素材。
- `/api/audio/search` 属于手板 AudioService、使用其独立检索与兜底缓存；量产 SearchPrescriptionAudio 属于 Somni AudioCatalogService，额外对照量产Mongo真值做状态/引用/阶段复核。这两条接口职责和数据源不同，不把手板HTTP结果当作可下发的处方准入结果；本次不将手板HTTP改连量产Mongo。共享检索的偏好排除仍共用修正后的解析规则。
- 人声内容分类合并到 audio_tags，依据 voice_guidance 的当前子树、lyrical_song、non_lexical_vocal；缺失/非法内容证据返回unknown，“不要人声”时保守排除。GetAudio 的 is_voice 保持既有声学指标 voice_presence_ratio>0 的显示语义，与“内容类别属于人声”是不同证据，**不能用该显示字段授权人声禁忌过滤**。本次不改变公开字段语义；声学检测误报/与内容标签冲突应另行核对素材。lullaby_style仅描述风格，不能证明含人声；是否需重新标注由素材/产品确认，不自动增加或去掉人声标签。
- 导出脚本使用 SOMNI_MONGO_DB（默认Somni），与应用一致；连接URI可不带库名。优先明确Somni URI，绝不回退Fullive。
- canonical_tag缓存当前内容和历史非内容身份集合；历史播放v6所用109项分类另存 audio_tag_aliases_v6.json，仅用于历史冻结语义，不作为新素材准入白名单。词典变更需重启服务；此次未部署。

---

# 2026-10-30 历史记录（标签快照已被本次修复替代）

以下记录属于2026-10-30的开发，基线为当时的 `origin/dev` `2ef51fc`，不代表本次分支基线。标签数量、旧编码及工程人声判定均为历史状态，当前以本文前半部分为准。

## 标签与检索

当时打包的 `app/core/audio_tag_aliases.json` 快照保留125条记录；后续确认其遗漏了数据库中的启用内容编码，因此不能继续作为当前权威字典。`worry_postponement`、`birdsong`、`fan_sound` 等新代码可用于对应维度的查询；不能把工程标签当作内容形态传入 `content_tag_codes`。

`audio_tag_aliases_v2.json` 只提供仍有效身份的旧显示名兼容；已取消类别不能靠旧字典重新变成有效的新查询代码。`voice` 保留为应用层人声排除条件，不虚构词典记录。

偏好规则使用当前类别。无对应类别的自由输入按 dev 既有逻辑保留诊断信息，不随意猜测成其他类别，也不自动改成静音。明确的静音选择继续生效。

素材的 `mechanism_tags`、`medical_risk_tags` 不再用于分类、排除或准入；全局词典中仍存在的同名领域不属于素材删字段范围。没有医学标签不表示素材已经通过医学安全评估。

## 素材字段

`SearchPrescriptionAudio` 保持原 RPC 与动态 `Struct` 响应，不新增 Proto 字段。返回素材保留：

- 内容形态、睡眠阶段、证据等级标签及原有 ID、父级关系。
- 完整工程标签，包括 `value`、`band_values`、`relative_loudness` 及实际提供的子项信息。
- 实际存在的语言、来源、描述、封面、时间及时长。
- `acoustic_metrics` 原值，包括 `duration_seconds`、`tempo_bpm`、`transient_ratio`、`transient_max_delta_db`、`voice_presence_ratio`。BPM 可以是 null；缺失指标不补零，不新增筛选阈值。

处方检索从 ES/缓存召回后，以 Mongo 中的当前启用状态、链接和标签为准。停用、缺失或标签已经变化的素材不放行；不再比较已废弃字段。工程标签变化仍会使旧候选失效。人声排除继续检查工程标签的具体值，不能把维度名称当作“有人声”。

本次实际数据库只读检查得到 200 条素材，均未提供 `acoustic_metrics`；对应代码已通过含完整指标和缺失指标的样例验证。当前真实数据缺失的数值不会凭空出现。

## 缓存与跨服务校验

检索结果与阶段索引使用标签契约版本；阶段素材文档和 GetAudio 目录缓存也加入版本，避免复用旧字典的结果。旧缓存自然过期，不需要修改数据库或清空远端 Redis。

以下命令仅校验，不修改 DeepHealth：

```bash
PYTHONPATH=. python scripts/export_audio_tag_contract.py --mirror-root /path/to/DeepHealth
```

校验当前字典、偏好规则、旧显示名兼容资产以及共享机器标签识别函数。两端对未知用户自由文本的业务策略可以不同；同步工具不会覆盖 DeepHealth 的完整解析模块。显式 `--write` 只同步字典资产，身份识别代码不一致时须先人工适配。

## 验证与发布边界

音频、检索、缓存、ES 和相关 RPC 回归共 336 项通过，其中包括 34 项检索任务经本机 gRPC 的完整编解码调用。测试使用真实业务处理代码和模拟查询依赖，不代表远端中间件已升级。

全量测试 831 项通过、6 项失败；这 6 项在未修改的 dev 上同样失败，分别为 5 项日志断言和 1 项英文词形处理。dev 对照共 11 项失败，本次修复了其中 5 项音频相关失败。

本次没有修改 Proto、数据库数据或部署配置，没有重建向量、刷新用户画像或下发方案。完成代码发布与部署后，再通过真实 `SearchPrescriptionAudio` 核验远端行为。DeepHealth 入站 `DecodeError` 是另一问题，本次不声称已经修复。
