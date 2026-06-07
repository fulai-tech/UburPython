#!/usr/bin/env python3
"""向 ES 写入检索回归测试用假数据（仅索引，不写 Mongo / 不调 comm）。

每条音频的扁平标签会经 EsSync 向量化并写入 tag_vectors（同名 label 复用已有向量）。

睡眠阶段仅允许：放松、入睡、守护、清醒（前缀 sleep:，如 sleep:放松）。

用法（项目根目录，已激活 .venv）:
  .venv/bin/python scripts/seed_es_test_data.py              # 默认灌 300 条
  .venv/bin/python scripts/seed_es_test_data.py --count 300
  .venv/bin/python scripts/seed_es_test_data.py --no-reseed # 不先清旧 seed 再写入
  .venv/bin/python scripts/seed_es_test_data.py --verify  # 写入后跑检索回归用例
  .venv/bin/python scripts/seed_es_test_data.py --export docs/seed-test-data-300.json --count 300

依赖: .env 中 ES_NODE、EMBEDDING_MODEL；首次会下载向量模型。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from elasticsearch import AsyncElasticsearch
from loguru import logger

from app.core.config import get_settings
from app.embedding.encoder import Encoder
from app.es.search import EsSearch
from app.es.sync import EsSync
from app.schemas.audio import SearchAudioRequest
from app.services.retrieval import RetrievalService

SEED_ID_PREFIX = "seed_audio_"
SEED_TOTAL_DEFAULT = 300
_GEN_RANDOM = random.Random(20260604)

# 业务约定：睡眠阶段仅此四值
SLEEP_STAGES = ("放松", "入睡", "守护", "清醒")


def _sleep(stage: str) -> str:
    if stage not in SLEEP_STAGES:
        raise ValueError(f"非法睡眠阶段: {stage}")
    return f"sleep:{stage}"


def _content(label: str) -> str:
    return f"content:{label}"


def _mechanism(label: str) -> str:
    return f"mechanism:{label}"


def _feat(label: str) -> str:
    return f"feat:{label}"


def _rhythm(label: str) -> str:
    return f"rhythm:{label}"


# 48 条锚点：固定名称供 eval 回归；其余由 build_seed_fixtures 程序化补齐至 300
_CURATED_FIXTURES: list[dict[str, object]] = [
    # —— 放松 ——
    {
        "id": f"{SEED_ID_PREFIX}001",
        "audio_url": "https://cdn.example.com/seed/relax-rain.mp3",
        "audio_name": "【测试】放松·深夜雨声",
        "flat_tags": [_sleep("放松"), _content("雨声"), _feat("低频持续")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}002",
        "audio_url": "https://cdn.example.com/seed/relax-forest.mp3",
        "audio_name": "【测试】放松·森林白噪音",
        "flat_tags": [_sleep("放松"), _content("森林"), _content("白噪音")],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    {
        "id": f"{SEED_ID_PREFIX}003",
        "audio_url": "https://cdn.example.com/seed/relax-stream.mp3",
        "audio_name": "【测试】放松·山间溪流",
        "flat_tags": [_sleep("放松"), _content("溪流"), _mechanism("注意力转移")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    {
        "id": f"{SEED_ID_PREFIX}004",
        "audio_url": "https://cdn.example.com/seed/relax-piano.mp3",
        "audio_name": "【测试】放松·钢琴轻音乐",
        "flat_tags": [_sleep("放松"), _content("钢琴"), _content("轻音乐")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}005",
        "audio_url": "https://cdn.example.com/seed/relax-fire.mp3",
        "audio_name": "【测试】放松·篝火白噪",
        "flat_tags": [_sleep("放松"), _content("篝火"), _rhythm("缓慢")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    {
        "id": f"{SEED_ID_PREFIX}006",
        "audio_url": "https://cdn.example.com/seed/relax-lavender.mp3",
        "audio_name": "【测试】放松·薰衣草呼吸引导",
        "flat_tags": [_sleep("放松"), _content("芳香疗愈"), _mechanism("腹式呼吸")],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    # —— 入睡 ——
    {
        "id": f"{SEED_ID_PREFIX}007",
        "audio_url": "https://cdn.example.com/seed/sleep-rain-fuzzy.mp3",
        "audio_name": "【测试】入睡·下雨的声音",
        "flat_tags": [_sleep("入睡"), _content("下雨的声音")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}008",
        "audio_url": "https://cdn.example.com/seed/sleep-meditation.mp3",
        "audio_name": "【测试】入睡·冥想音乐",
        "flat_tags": [_sleep("入睡"), _content("冥想音乐"), _mechanism("正念")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    {
        "id": f"{SEED_ID_PREFIX}009",
        "audio_url": "https://cdn.example.com/seed/sleep-432hz.mp3",
        "audio_name": "【测试】入睡·432Hz助眠",
        "flat_tags": [_sleep("入睡"), _content("纯音"), _feat("432Hz")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}010",
        "audio_url": "https://cdn.example.com/seed/sleep-lullaby.mp3",
        "audio_name": "【测试】入睡·摇篮曲",
        "flat_tags": [_sleep("入睡"), _content("摇篮曲"), _rhythm("轻柔摇摆")],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    {
        "id": f"{SEED_ID_PREFIX}011",
        "audio_url": "https://cdn.example.com/seed/sleep-breath.mp3",
        "audio_name": "【测试】入睡·数息引导",
        "flat_tags": [_sleep("入睡"), _content("数息"), _mechanism("渐进放松")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    {
        "id": f"{SEED_ID_PREFIX}012",
        "audio_url": "https://cdn.example.com/seed/sleep-ocean.mp3",
        "audio_name": "【测试】入睡·海浪轻抚",
        "flat_tags": [_sleep("入睡"), _content("海浪")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    # —— 守护 ——
    {
        "id": f"{SEED_ID_PREFIX}013",
        "audio_url": "https://cdn.example.com/seed/guard-white-noise.mp3",
        "audio_name": "【测试】守护·整夜白噪声",
        "flat_tags": [_sleep("守护"), _content("白噪音"), _feat("掩蔽突发声")],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    {
        "id": f"{SEED_ID_PREFIX}014",
        "audio_url": "https://cdn.example.com/seed/guard-hum.mp3",
        "audio_name": "【测试】守护·母亲哼唱",
        "flat_tags": [_sleep("守护"), _content("哼唱"), _mechanism("安全感")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}015",
        "audio_url": "https://cdn.example.com/seed/guard-fetal.mp3",
        "audio_name": "【测试】守护·胎心节律",
        "flat_tags": [_sleep("守护"), _content("胎心音"), _rhythm("稳定脉冲")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    {
        "id": f"{SEED_ID_PREFIX}016",
        "audio_url": "https://cdn.example.com/seed/guard-pendulum.mp3",
        "audio_name": "【测试】守护·钟摆滴答",
        "flat_tags": [_sleep("守护"), _content("钟摆"), _feat("规律滴答")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    {
        "id": f"{SEED_ID_PREFIX}017",
        "audio_url": "https://cdn.example.com/seed/guard-story.mp3",
        "audio_name": "【测试】守护·夜间故事",
        "flat_tags": [_sleep("守护"), _content("故事叙述"), _mechanism("陪伴")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}018",
        "audio_url": "https://cdn.example.com/seed/guard-dual-heart.mp3",
        "audio_name": "【测试】守护·双心跳",
        "flat_tags": [_sleep("守护"), _content("心跳"), _rhythm("双节拍")],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    # —— 清醒 ——
    {
        "id": f"{SEED_ID_PREFIX}019",
        "audio_url": "https://cdn.example.com/seed/wake-birds.mp3",
        "audio_name": "【测试】清醒·晨间鸟鸣",
        "flat_tags": [_sleep("清醒"), _content("鸟鸣"), _content("自然")],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    {
        "id": f"{SEED_ID_PREFIX}020",
        "audio_url": "https://cdn.example.com/seed/wake-alarm.mp3",
        "audio_name": "【测试】清醒·渐进闹钟",
        "flat_tags": [_sleep("清醒"), _content("闹钟"), _feat("渐强")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}021",
        "audio_url": "https://cdn.example.com/seed/wake-sun.mp3",
        "audio_name": "【测试】清醒·阳光冥想",
        "flat_tags": [_sleep("清醒"), _content("日光冥想")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    {
        "id": f"{SEED_ID_PREFIX}022",
        "audio_url": "https://cdn.example.com/seed/wake-rock.mp3",
        "audio_name": "【测试】清醒·活力摇滚",
        "flat_tags": [_sleep("清醒"), _content("摇滚"), _content("嘈杂")],
        "evidence_level": "D",
        "recommend_weight": 0.2,
    },
    {
        "id": f"{SEED_ID_PREFIX}023",
        "audio_url": "https://cdn.example.com/seed/wake-news.mp3",
        "audio_name": "【测试】清醒·简报播客",
        "flat_tags": [_sleep("清醒"), _content("播客"), _mechanism("信息唤醒")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    {
        "id": f"{SEED_ID_PREFIX}024",
        "audio_url": "https://cdn.example.com/seed/wake-coffee.mp3",
        "audio_name": "【测试】清醒·咖啡店环境",
        "flat_tags": [_sleep("清醒"), _content("咖啡店"), _content("人声环境")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    # —— 放松：多标签重叠（粗排 / 多标签召回）——
    {
        "id": f"{SEED_ID_PREFIX}025",
        "audio_url": "https://cdn.example.com/seed/relax-nature-mix.mp3",
        "audio_name": "【测试】放松·自然四合一",
        "flat_tags": [
            _sleep("放松"),
            _content("雨声"),
            _content("森林"),
            _content("白噪音"),
            _content("溪流"),
        ],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    {
        "id": f"{SEED_ID_PREFIX}026",
        "audio_url": "https://cdn.example.com/seed/relax-piano-rain.mp3",
        "audio_name": "【测试】放松·钢琴雨夜",
        "flat_tags": [_sleep("放松"), _content("雨声"), _content("钢琴"), _content("轻音乐")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}027",
        "audio_url": "https://cdn.example.com/seed/relax-forest-stream.mp3",
        "audio_name": "【测试】放松·林溪篝火",
        "flat_tags": [_sleep("放松"), _content("森林"), _content("溪流"), _content("篝火")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    {
        "id": f"{SEED_ID_PREFIX}028",
        "audio_url": "https://cdn.example.com/seed/relax-aroma-music.mp3",
        "audio_name": "【测试】放松·芳香轻音乐",
        "flat_tags": [
            _sleep("放松"),
            _content("芳香疗愈"),
            _content("轻音乐"),
            _mechanism("腹式呼吸"),
        ],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}029",
        "audio_url": "https://cdn.example.com/seed/relax-white-rain.mp3",
        "audio_name": "【测试】放松·白噪雨声",
        "flat_tags": [_sleep("放松"), _content("白噪音"), _content("雨声"), _feat("掩蔽突发声")],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    {
        "id": f"{SEED_ID_PREFIX}030",
        "audio_url": "https://cdn.example.com/seed/relax-meditation-soft.mp3",
        "audio_name": "【测试】放松·冥想轻音乐",
        "flat_tags": [_sleep("放松"), _content("冥想音乐"), _content("轻音乐"), _mechanism("正念")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    # —— 入睡：多标签 ——
    {
        "id": f"{SEED_ID_PREFIX}031",
        "audio_url": "https://cdn.example.com/seed/sleep-ocean-rain.mp3",
        "audio_name": "【测试】入睡·海雨助眠",
        "flat_tags": [_sleep("入睡"), _content("海浪"), _content("雨声"), _content("摇篮曲")],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    {
        "id": f"{SEED_ID_PREFIX}032",
        "audio_url": "https://cdn.example.com/seed/sleep-meditation-breath.mp3",
        "audio_name": "【测试】入睡·冥想数息",
        "flat_tags": [
            _sleep("入睡"),
            _content("冥想音乐"),
            _content("数息"),
            _mechanism("渐进放松"),
        ],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}033",
        "audio_url": "https://cdn.example.com/seed/sleep-lullaby-ocean.mp3",
        "audio_name": "【测试】入睡·海浪摇篮",
        "flat_tags": [_sleep("入睡"), _content("海浪"), _content("摇篮曲"), _rhythm("轻柔摇摆")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}034",
        "audio_url": "https://cdn.example.com/seed/sleep-pure-tone.mp3",
        "audio_name": "【测试】入睡·纯音海浪",
        "flat_tags": [_sleep("入睡"), _content("纯音"), _content("海浪"), _feat("432Hz")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    {
        "id": f"{SEED_ID_PREFIX}035",
        "audio_url": "https://cdn.example.com/seed/sleep-rain-lullaby.mp3",
        "audio_name": "【测试】入睡·雨声摇篮",
        "flat_tags": [_sleep("入睡"), _content("下雨的声音"), _content("摇篮曲"), _content("哼唱")],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    {
        "id": f"{SEED_ID_PREFIX}036",
        "audio_url": "https://cdn.example.com/seed/sleep-breath-mindful.mp3",
        "audio_name": "【测试】入睡·正念呼吸",
        "flat_tags": [_sleep("入睡"), _content("数息"), _mechanism("正念"), _mechanism("渐进放松")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    # —— 守护：多标签 ——
    {
        "id": f"{SEED_ID_PREFIX}037",
        "audio_url": "https://cdn.example.com/seed/guard-noise-heart.mp3",
        "audio_name": "【测试】守护·白噪心跳",
        "flat_tags": [_sleep("守护"), _content("白噪音"), _content("心跳"), _content("哼唱")],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    {
        "id": f"{SEED_ID_PREFIX}038",
        "audio_url": "https://cdn.example.com/seed/guard-story-hum.mp3",
        "audio_name": "【测试】守护·故事哼唱",
        "flat_tags": [
            _sleep("守护"),
            _content("故事叙述"),
            _content("哼唱"),
            _mechanism("陪伴"),
        ],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}039",
        "audio_url": "https://cdn.example.com/seed/guard-fetal-pendulum.mp3",
        "audio_name": "【测试】守护·胎心钟摆",
        "flat_tags": [_sleep("守护"), _content("胎心音"), _content("钟摆"), _rhythm("稳定脉冲")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    {
        "id": f"{SEED_ID_PREFIX}040",
        "audio_url": "https://cdn.example.com/seed/guard-triple-calm.mp3",
        "audio_name": "【测试】守护·三重安抚",
        "flat_tags": [
            _sleep("守护"),
            _content("白噪音"),
            _content("心跳"),
            _content("哼唱"),
            _mechanism("安全感"),
        ],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    # —— 清醒：多标签 ——
    {
        "id": f"{SEED_ID_PREFIX}041",
        "audio_url": "https://cdn.example.com/seed/wake-nature-sun.mp3",
        "audio_name": "【测试】清醒·鸟鸣日光",
        "flat_tags": [_sleep("清醒"), _content("鸟鸣"), _content("自然"), _content("日光冥想")],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    {
        "id": f"{SEED_ID_PREFIX}042",
        "audio_url": "https://cdn.example.com/seed/wake-cafe-podcast.mp3",
        "audio_name": "【测试】清醒·咖啡播客",
        "flat_tags": [_sleep("清醒"), _content("咖啡店"), _content("播客"), _content("人声环境")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}043",
        "audio_url": "https://cdn.example.com/seed/wake-alarm-energy.mp3",
        "audio_name": "【测试】清醒·闹钟摇滚",
        "flat_tags": [_sleep("清醒"), _content("闹钟"), _content("摇滚"), _feat("渐强")],
        "evidence_level": "D",
        "recommend_weight": 0.2,
    },
    {
        "id": f"{SEED_ID_PREFIX}044",
        "audio_url": "https://cdn.example.com/seed/wake-morning-mix.mp3",
        "audio_name": "【测试】清醒·晨间四合一",
        "flat_tags": [
            _sleep("清醒"),
            _content("鸟鸣"),
            _content("自然"),
            _content("日光冥想"),
            _content("播客"),
        ],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
    {
        "id": f"{SEED_ID_PREFIX}045",
        "audio_url": "https://cdn.example.com/seed/wake-noisy-cafe.mp3",
        "audio_name": "【测试】清醒·嘈杂咖啡",
        "flat_tags": [_sleep("清醒"), _content("咖啡店"), _content("嘈杂"), _content("人声环境")],
        "evidence_level": "C",
        "recommend_weight": 0.45,
    },
    {
        "id": f"{SEED_ID_PREFIX}046",
        "audio_url": "https://cdn.example.com/seed/wake-podcast-news.mp3",
        "audio_name": "【测试】清醒·播客简报",
        "flat_tags": [_sleep("清醒"), _content("播客"), _mechanism("信息唤醒")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}047",
        "audio_url": "https://cdn.example.com/seed/relax-triple-noise.mp3",
        "audio_name": "【测试】放松·三重白噪",
        "flat_tags": [_sleep("放松"), _content("白噪音"), _content("篝火"), _content("森林")],
        "evidence_level": "B",
        "recommend_weight": 0.75,
    },
    {
        "id": f"{SEED_ID_PREFIX}048",
        "audio_url": "https://cdn.example.com/seed/sleep-quad-calm.mp3",
        "audio_name": "【测试】入睡·四重助眠",
        "flat_tags": [
            _sleep("入睡"),
            _content("海浪"),
            _content("雨声"),
            _content("摇篮曲"),
            _content("哼唱"),
        ],
        "evidence_level": "A",
        "recommend_weight": 1.0,
    },
]

# 各睡眠阶段标签池（程序化生成 049+ 时复用，保证多标签重叠）
_STAGE_TAG_POOLS: dict[str, dict[str, list[str]]] = {
    "放松": {
        "content": [
            "雨声",
            "森林",
            "白噪音",
            "溪流",
            "钢琴",
            "轻音乐",
            "篝火",
            "芳香疗愈",
            "冥想音乐",
            "自然",
            "风铃",
            "钵声",
        ],
        "mechanism": ["注意力转移", "腹式呼吸", "正念", "渐进放松"],
        "feat": ["低频持续", "掩蔽突发声", "立体声"],
        "rhythm": ["缓慢", "均匀"],
    },
    "入睡": {
        "content": [
            "下雨的声音",
            "冥想音乐",
            "海浪",
            "摇篮曲",
            "数息",
            "哼唱",
            "纯音",
            "432Hz",
            "雨声",
            "白噪音",
            "深度睡眠",
        ],
        "mechanism": ["渐进放松", "正念", "数息法"],
        "feat": ["432Hz", "Delta波", "Theta波"],
        "rhythm": ["轻柔摇摆", "渐弱"],
    },
    "守护": {
        "content": [
            "白噪音",
            "哼唱",
            "心跳",
            "胎心音",
            "钟摆",
            "故事叙述",
            "陪伴",
            "安全感",
            "夜间巡视",
        ],
        "mechanism": ["安全感", "陪伴", "稳定节律"],
        "feat": ["规律滴答", "掩蔽突发声"],
        "rhythm": ["稳定脉冲", "双节拍"],
    },
    "清醒": {
        "content": [
            "鸟鸣",
            "自然",
            "日光冥想",
            "摇滚",
            "嘈杂",
            "咖啡店",
            "播客",
            "闹钟",
            "人声环境",
            "简报",
            "活力音乐",
        ],
        "mechanism": ["信息唤醒", "渐进唤醒"],
        "feat": ["渐强", "高频明亮"],
        "rhythm": ["轻快", "律动"],
    },
}

_GEN_THEMES = ("自然", "疗愈", "精选", "混合", "舒缓", "深度", "轻量", "定制")
_EVIDENCE_LEVELS = ("A", "B", "C", "D")
_EVIDENCE_WEIGHT = {"A": 1.0, "B": 0.75, "C": 0.45, "D": 0.2}


def _generate_fixtures(from_seq: int, to_seq: int) -> list[dict[str, object]]:
    """程序化生成 seed_audio_{from_seq}..{to_seq}，每阶段均匀分布、2~5 个标签。"""
    rows: list[dict[str, object]] = []
    for seq in range(from_seq, to_seq + 1):
        stage = SLEEP_STAGES[(seq - 1) % len(SLEEP_STAGES)]
        pool = _STAGE_TAG_POOLS[stage]
        flat: list[str] = [_sleep(stage)]

        n_content = 2 + (seq % 3)
        content_pick = _GEN_RANDOM.sample(pool["content"], min(n_content, len(pool["content"])))
        flat.extend(_content(c) for c in content_pick)

        if seq % 2 == 0:
            flat.append(_mechanism(_GEN_RANDOM.choice(pool["mechanism"])))
        if seq % 3 == 0:
            flat.append(_feat(_GEN_RANDOM.choice(pool["feat"])))
        if seq % 5 == 0:
            flat.append(_rhythm(_GEN_RANDOM.choice(pool["rhythm"])))

        ev = _EVIDENCE_LEVELS[seq % len(_EVIDENCE_LEVELS)]
        theme = _GEN_THEMES[seq % len(_GEN_THEMES)]
        rows.append(
            {
                "id": f"{SEED_ID_PREFIX}{seq:03d}",
                "audio_url": f"https://cdn.example.com/seed/auto-{seq:03d}.mp3",
                "audio_name": f"【测试】{stage}·{theme}{seq:03d}",
                "flat_tags": flat,
                "evidence_level": ev,
                "recommend_weight": _EVIDENCE_WEIGHT[ev],
            }
        )
    return rows


def build_seed_fixtures(total: int = SEED_TOTAL_DEFAULT) -> list[dict[str, object]]:
    if total <= len(_CURATED_FIXTURES):
        return _CURATED_FIXTURES[:total]
    extra = _generate_fixtures(len(_CURATED_FIXTURES) + 1, total)
    return _CURATED_FIXTURES + extra


def export_seed_fixtures_json(path: Path, total: int = SEED_TOTAL_DEFAULT) -> None:
    """将 seed 假数据导出为 JSON（与写入 ES 的字段一致，不连 ES）。"""
    fixtures = build_seed_fixtures(total)
    stage_counts: dict[str, int] = {s: 0 for s in SLEEP_STAGES}
    records: list[dict[str, object]] = []
    for row in fixtures:
        flat_tags = list(row["flat_tags"])  # type: ignore[arg-type]
        for tag in flat_tags:
            if tag.startswith("sleep:"):
                stage_counts[tag.removeprefix("sleep:")] += 1
                break
        records.append(
            {
                "id": row["id"],
                "audio_url": row["audio_url"],
                "audio_name": row["audio_name"],
                "flat_tags": flat_tags,
                "evidence_level": row["evidence_level"],
                "recommend_weight": row["recommend_weight"],
            }
        )
    payload = {
        "count": len(records),
        "source": "scripts/seed_es_test_data.py",
        "sleep_stage_counts": stage_counts,
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("已导出 {} 条测试数据 → {}", len(records), path)


@dataclass(frozen=True)
class SearchCase:
    name: str
    request: SearchAudioRequest
    expect_names: list[str]
    note: str


SEARCH_CASES: list[SearchCase] = [
    SearchCase(
        name="放松+雨声精确",
        request=SearchAudioRequest(
            sleep_stage_tags=["放松"],
            content_tags=["雨声"],
            top_k=10,
        ),
        expect_names=["【测试】放松·深夜雨声"],
        note="001 精确；跨阶段 007 为入睡+下雨的声音（睡眠过滤后不在候选）",
    ),
    SearchCase(
        name="入睡+雨声向量模糊",
        request=SearchAudioRequest(
            sleep_stage_tags=["入睡"],
            content_tags=["雨声"],
            top_k=10,
        ),
        expect_names=["【测试】入睡·下雨的声音"],
        note="007 标签为「下雨的声音」，靠 tag_vectors 与「雨声」向量相似命中",
    ),
    SearchCase(
        name="放松+森林白噪音双标签",
        request=SearchAudioRequest(
            sleep_stage_tags=["放松"],
            content_tags=["森林", "白噪音"],
            top_k=10,
        ),
        expect_names=["【测试】放松·森林白噪音"],
        note="002 match_count=2",
    ),
    SearchCase(
        name="入睡阶段查海浪",
        request=SearchAudioRequest(
            sleep_stage_tags=["入睡"],
            content_tags=["海浪"],
            top_k=10,
        ),
        expect_names=["【测试】入睡·海浪轻抚"],
        note="012；放松/守护/清醒 带海浪的应被睡眠过滤掉",
    ),
    SearchCase(
        name="守护仅睡眠阶段",
        request=SearchAudioRequest(sleep_stage_tags=["守护"], top_k=100),
        expect_names=[
            "【测试】守护·整夜白噪声",
            "【测试】守护·母亲哼唱",
            "【测试】守护·胎心节律",
            "【测试】守护·钟摆滴答",
            "【测试】守护·夜间故事",
            "【测试】守护·双心跳",
        ],
        note="锚点 6 条应包含在守护阶段全量结果中（300 条时守护约 75 条）",
    ),
    SearchCase(
        name="清醒+厌恶嘈杂",
        request=SearchAudioRequest(
            sleep_stage_tags=["清醒"],
            content_tags=["摇滚"],
            disliked_tags=["嘈杂"],
            top_k=10,
        ),
        expect_names=[],
        note="022 含摇滚与嘈杂，应被厌恶剔除",
    ),
    SearchCase(
        name="入睡+冥想向量模糊",
        request=SearchAudioRequest(
            sleep_stage_tags=["入睡"],
            content_tags=["冥想"],
            top_k=10,
        ),
        expect_names=["【测试】入睡·冥想音乐"],
        note="008 精确或模糊「冥想」↔「冥想音乐」",
    ),
]


def _print_manual_curl(base_url: str) -> None:
    cases = [
        {
            "sleep_stage_tags": ["放松"],
            "content_tags": ["雨声"],
            "disliked_tags": [],
            "top_k": 10,
        },
        {
            "sleep_stage_tags": ["入睡"],
            "content_tags": ["雨声"],
            "disliked_tags": [],
            "top_k": 10,
        },
        {
            "sleep_stage_tags": ["守护"],
            "content_tags": ["白噪音"],
            "disliked_tags": [],
            "top_k": 10,
        },
        {
            "sleep_stage_tags": ["清醒"],
            "content_tags": ["鸟鸣", "自然"],
            "disliked_tags": [],
            "top_k": 10,
        },
        {
            "sleep_stage_tags": ["清醒"],
            "content_tags": ["摇滚"],
            "disliked_tags": ["嘈杂"],
            "top_k": 10,
        },
    ]
    print("\n--- Apifox / curl 检索示例（睡眠阶段：放松/入睡/守护/清醒）---\n")
    for idx, body in enumerate(cases, start=1):
        print(f"# 用例 {idx}")
        print(
            f"curl -s -X POST '{base_url}/api/audio/search' "
            f"-H 'Content-Type: application/json' "
            f"-d '{json.dumps(body, ensure_ascii=False)}' | python -m json.tool\n"
        )


async def _clear_seed_audio(client: AsyncElasticsearch, audio_index: str) -> int:
    """按 audio_name 前缀清理测试数据（_id 字段不支持 prefix 查询）。"""
    response = await client.delete_by_query(
        index=audio_index,
        body={"query": {"prefix": {"audio_name": "【测试】"}}},
        refresh=True,
    )
    return int(response.get("deleted", 0))


async def _count_tag_vectors(client: AsyncElasticsearch, index: str) -> int:
    response = await client.count(index=index)
    return int(response["count"])


async def _seed(settings, *, verify: bool, reseed: bool, total: int) -> None:
    client = AsyncElasticsearch(settings.es_node)
    encoder = Encoder(settings)
    encoder.load()
    es_search = EsSearch(client, settings)
    await es_search.ensure_indices()
    es_sync = EsSync(client, encoder, settings)
    fixtures = build_seed_fixtures(total)

    if reseed:
        deleted = await _clear_seed_audio(client, settings.es_audio_index)
        logger.info("已清理旧 seed 音频 {} 条，即将重新写入", deleted)

    tag_count_before = await _count_tag_vectors(client, settings.es_tag_vectors_index)

    for idx, row in enumerate(fixtures, start=1):
        await es_sync.upsert_audio(
            str(row["id"]),
            audio_url=str(row["audio_url"]),
            audio_name=str(row["audio_name"]),
            flat_tags=list(row["flat_tags"]),  # type: ignore[arg-type]
            evidence_level=str(row["evidence_level"]),
            recommend_weight=float(row["recommend_weight"]),  # type: ignore[arg-type]
        )
        if idx % 50 == 0 or idx == len(fixtures):
            logger.info("写入进度 {}/{}", idx, len(fixtures))

    tag_count_after = await _count_tag_vectors(client, settings.es_tag_vectors_index)
    logger.info(
        "已写入 {} 条测试音频 → {}；tag_vectors 文档数 {} → {}（标签已向量化入库）",
        len(fixtures),
        settings.es_audio_index,
        tag_count_before,
        tag_count_after,
    )

    if verify:
        retrieval = RetrievalService(es_search, encoder, settings)
        print("\n--- 内置检索回归（进程内 RetrievalService）---\n")
        for case in SEARCH_CASES:
            results = await retrieval.search(case.request)
            names = [r.audio_name for r in results]
            expect_set = set(case.expect_names)
            hit_set = set(names)
            ok = expect_set.issubset(hit_set) if case.expect_names else len(names) == 0
            status = "OK" if ok else "CHECK"
            print(f"[{status}] {case.name}: 命中={names}")
            print(f"       说明: {case.note}\n")

    _print_manual_curl(f"http://{settings.app_host}:{settings.app_port}")
    await client.close()


async def _main() -> None:
    parser = argparse.ArgumentParser(description="ES 检索测试数据种子")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="仅删除 id 以 seed_audio_ 开头的 audio_materials 文档",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=SEED_TOTAL_DEFAULT,
        help=f"写入条数（默认 {SEED_TOTAL_DEFAULT}，前 48 条为固定锚点）",
    )
    parser.add_argument(
        "--no-reseed",
        action="store_true",
        help="不先清理旧 seed_audio_* 文档（默认会先清理再写入）",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="写入后执行内置检索用例（需加载 embedding 模型）",
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="PATH",
        help="仅导出 seed 假数据到 JSON 文件（不连 ES、不写入）",
    )
    args = parser.parse_args()

    if args.export:
        export_seed_fixtures_json(args.export, total=args.count)
        return

    settings = get_settings()
    client = AsyncElasticsearch(settings.es_node)
    try:
        if args.clear:
            deleted = await _clear_seed_audio(client, settings.es_audio_index)
            logger.info("已清理 seed 音频文档 {} 条（tag_vectors 保留）", deleted)
            return
        await _seed(settings, verify=args.verify, reseed=not args.no_reseed, total=args.count)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(_main())
