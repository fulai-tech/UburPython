# 画像数据边界与来源摘要

## 首次与更新

`profile_update` 显式 `initial` 只读问卷、既有画像状态、客观生理摘要和睡眠债。未指定用途时先仅探测既有画像：不存在才走首次；存在则走刷新。两次读取之间画像状态变化返回可重试错误，不把不确定状态当作首次。已有画像显式 initial 仍拒绝。

中间件 initial 的分区白名单为 `questionnaires / prior_profile / facts / sleep_debt`；请求其他分区返回 `INVALID_ARGUMENT: initial_section_not_allowed`。更新、纠正、周复盘沿用既有用途与参数，不新增 update 字段。

## 生理摘要

只投影可穿戴按日指标、Somni 整夜汇总及报告指标，不拉取传感器点、完整 raw_data 数组、上传记录 ID 或逐点时间窗。Somni 历史记录的 raw_data 下仅读取已用于整夜算法的白名单标量（起止时间、睡眠比例、潜伏期、效率），不会在线从原始点计算新摘要。无可确认的主睡眠、质量、版本或一致报告时标记缺失/冲突。

来源指向实际贡献指标的摘要记录，不展开其底层采集记录。同日中英文报告值一致时选确定的一份引用；冲突继续标记缺失。睡眠债仅引用自身参与计算的摘要，不能复制整个请求的全局 manifest。

## 偏好证据

中间件在完整有效窗口内去重、合并操作手势，并验证方案与会话；每个手势保留起点/终点的代表记录。操作观察同时关联会话和基准方案，避免只保留终值而失去上下文。未参与聚合的原始操作不进入 manifest，operation_bursts 只输出数量和摘要，不返回全部中间步骤。

DPH 仍按完整证据计算独立夜数、一致率、目标值及冲突。计算后才压缩画像：每个参数保留最近的正向、反向、中性/冲突夜，保留原值、终值、持续生效/反转状态、关键来源引用及全窗口统计。不能用保留的三夜样本重新推断全窗口偏好。

## JSON 与兼容

Proto、业务指标、安全字段、作息及长期画像内容 hash 不变。画像的 `based_on` 新增以下可选字段，调用方完整保存，不自行展开底层来源：

```json
{
  "source_refs": ["questionnaire_answer:answer-1"],
  "reference_table": [
    {"ref": "questionnaire_answer:answer-1", "source_type": "questionnaire_answer", "source_id": "answer-1"}
  ],
  "provenance_summary": {
    "version": "profile_provenance.v1",
    "input_reference_count": 1,
    "unique_reference_count": 1,
    "retained_reference_count": 1,
    "digest": "sha256:example-only",
    "compacted": true,
    "references_omitted": false
  }
}
```

画像内部 `source_refs` 为规范引用字符串；元信息仅放在该画像的 `based_on.reference_table`。旧画像中的对象引用和嵌套 source_manifest 在读取时转换，数据库原文不变。既有资料中的对象引用示例表示历史格式；方案/每日决策证据及 protobuf 顶层 SourceRef 不采用这次画像投影。

摘要包含输入出现次数、唯一引用数、保留数、摘要版本和确定性 hash；偏好 source_summary 另外包含扫描记录数、有效观察数及窗口。hash 不包含读取时新生成的 observed_at，也不把画像自身加入来源，保证同日复用。hash 用于对照，不代表可恢复完整历史。

最多保留 256 条不同直接引用，来源审计部分不超过 128 KiB；预算不足时继续减少引用并设置 references_omitted，不截断业务数据。已有大画像只精简来源/审计；返回视图重新计算 content_hash，长期画像链接 hash 保留。身份与链接错误仍拒绝，业务数据自身超出最终 protobuf 限额仍返回 RESOURCE_EXHAUSTED。

## 发布与验证

先审核并部署中间件，再部署 DPH。仅更新 DPH 不能修复旧大画像的远程读取；本地回放成功不等于远端已恢复。不迁移或重写历史画像/方案。回滚需要保留中间件的精简读取能力，否则旧大画像可能重新超过响应预算。

回归覆盖首次无操作查询、首次非法分区、摘要字段投影、多语言冲突、偏好计算不变、同日复用、旧画像体积、身份/hash 错误及每日确定性重放。
