"""loguru 初始化：控制台（开发）+ 按日文件（生产采集）。

文件落在 LOG_DIR，每日一个，命名为 YYYY-MM-DD_ubur_log（不隐藏、不压缩）。
enqueue=True：异步写盘，避免阻塞请求线程。
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from loguru import logger

from app.core.config import Settings

FILE_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} | {message}"
)

CONSOLE_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

LOG_FILE_SUFFIX = "_ubur_log"
_DAILY_ROTATION = "00:00"


def current_log_path(log_dir: Path, when: date | None = None) -> Path:
    """返回指定日期（默认今天）的日志文件路径：YYYY-MM-DD_ubur_log。"""
    day = when or date.today()
    return log_dir / f"{day:%Y-%m-%d}{LOG_FILE_SUFFIX}"


def setup_logging(settings: Settings) -> Path:
    """初始化双输出；返回当天日志文件路径供启动日志引用。"""
    log_dir = settings.log_dir_path
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = current_log_path(log_dir)
    sink_pattern = str(log_dir / f"{{time:YYYY-MM-DD}}{LOG_FILE_SUFFIX}")

    logger.remove()
    level = settings.log_level.upper()
    logger.add(sys.stderr, level=level, format=CONSOLE_LOG_FORMAT)
    logger.add(
        sink_pattern,
        level=level,
        format=FILE_LOG_FORMAT,
        rotation=_DAILY_ROTATION,
        retention=settings.log_retention,
        encoding="utf-8",
        enqueue=True,
    )
    logger.info("日志已初始化，级别={}，文件={}", level, log_file)
    return log_file
