"""loguru 初始化：控制台（开发）+ 按日/按大小文件（生产采集）。

文件落在 LOG_DIR，命名为 YYYY-MM-DD_ubur_log。
- 本地日历跨日：切到新日期文件。
- 单日文件超过 log_rotation_size：旧文件追加时间戳后缀归档，新建同名文件继续写。
enqueue=True：异步写盘，避免阻塞请求线程。
extra.request_id：并发请求可按 request_id 串联整条调用链日志。
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import IO, Any

from loguru import logger

from app.core.config import Settings

FILE_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{extra[request_id]} | {name}:{function}:{line} | {message}"
)

CONSOLE_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<yellow>{extra[request_id]}</yellow> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

DEFAULT_REQUEST_ID = "-"
LOG_FILE_SUFFIX = "_ubur_log"
_SIZE_PATTERN = re.compile(
    r"([eE+\-\.\d]+)\s*([kmgtpezyKMGTPEZY])?(i)?([bB])"
)


def current_log_path(log_dir: Path, when: date | None = None) -> Path:
    """返回指定日期（默认今天）的日志文件路径：YYYY-MM-DD_ubur_log。"""
    day = when or date.today()
    return log_dir / f"{day:%Y-%m-%d}{LOG_FILE_SUFFIX}"


def parse_log_size(size: str) -> int:
    """将 '100 MB' / '1 GiB' 等解析为字节数（规则与 loguru 一致）。"""
    match = _SIZE_PATTERN.fullmatch(size.strip())
    if match is None:
        raise ValueError(f"无法解析 log_rotation_size: {size!r}")
    number_text, unit, binary, byte_unit = match.groups()
    number = float(number_text)
    unit_power = "kmgtpezy".index(unit.lower()) + 1 if unit else 0
    base = 1024 if binary else 1000
    bit_div = 8 if byte_unit == "b" else 1
    return int(number * (base**unit_power) / bit_div)


def daily_or_size_rotation(max_bytes: int) -> Callable[[Any, IO[str]], bool]:
    """跨日或超过大小时滚动；同日满文件由 loguru 重命名旧文件后再开新文件。"""

    def should_rotate(message: Any, file: IO[str]) -> bool:
        file.seek(0, 2)
        if file.tell() + len(message) > max_bytes:
            return True
        file_day = Path(file.name).name[:10]
        msg_day = message.record["time"].strftime("%Y-%m-%d")
        return file_day != msg_day

    return should_rotate


def setup_logging(settings: Settings) -> Path:
    """初始化双输出；返回当天日志文件路径供启动日志引用。"""
    log_dir = settings.log_dir_path
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = current_log_path(log_dir)
    sink_pattern = str(log_dir / f"{{time:YYYY-MM-DD}}{LOG_FILE_SUFFIX}")
    max_bytes = parse_log_size(settings.log_rotation_size)

    logger.remove()
    logger.configure(extra={"request_id": DEFAULT_REQUEST_ID})
    level = settings.log_level.upper()
    logger.add(sys.stderr, level=level, format=CONSOLE_LOG_FORMAT)
    logger.add(
        sink_pattern,
        level=level,
        format=FILE_LOG_FORMAT,
        rotation=daily_or_size_rotation(max_bytes),
        retention=settings.log_retention,
        encoding="utf-8",
        enqueue=True,
    )
    logger.info(
        "日志已初始化，级别={}，文件={}，单文件上限={}",
        level,
        log_file,
        settings.log_rotation_size,
    )
    return log_file
