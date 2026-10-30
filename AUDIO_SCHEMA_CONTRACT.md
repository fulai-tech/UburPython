# 新版音频素材读取契约

本次开发以 `origin/dev` 的 `2ef51fc` 为基线。代码完成本地验证，尚未部署。

## 标签与检索

`app/core/audio_tag_aliases.json` 对应当前数据库词典：127 条中保留 125 条有效记录，2 条数字测试阶段不作为有效机器代码。`worry_postponement`、`birdsong`、`fan_sound` 等新代码可用于对应维度的查询；不能把工程标签当作内容形态传入 `content_tag_codes`。

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
