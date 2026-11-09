"""loguru 初始化：控制台 + 按日/按大小文件。

公共日志落在 LOG_DIR/YYYY-MM-DD_ubur_log（HTTP、启动、ES、编码器等）。
每个 gRPC 服务一个文件，访问日志和该服务自己的日志写在一起：

- logs/{handboard|somni}/{服务}/YYYY-MM-DD_log

路由规则：拦截器 bind 的 log_product/log_service，或模块名 app.server.{产品}.{服务}.*，
都进该服务的同一个日志文件。
- 本地日历跨日：切到新日期文件。
- 单日文件超过 log_rotation_size：旧文件追加时间戳后缀归档，新建同名文件继续写。
enqueue=True：异步写盘，避免阻塞请求线程。
extra.request_id：并发请求可按 request_id 串联整条调用链日志。
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from contextvars import ContextVar
from datetime import date
from pathlib import Path
from typing import IO, Any

from loguru import logger

from app.core.config import Settings

FILE_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{extra[request_id]} | {name}:{function}:{line} | {message}"
)
# tail -f 直接显示这些 ANSI 码。正常为绿，警告为黄，错误为红。
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"
_FILE_LEVEL_COLOR = {
    "WARNING": _ANSI_YELLOW,
    "ERROR": _ANSI_RED,
    "CRITICAL": _ANSI_RED,
}

CONSOLE_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<yellow>{extra[request_id]}</yellow> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

DEFAULT_REQUEST_ID = "-"
LOG_FILE_SUFFIX = "_ubur_log"
LAYER_CONTROLLER = "controller"
LAYER_SERVICE = "service"
LAYER_LOG = "log"
_MODULE_RE = re.compile(
    r"^app\.server\.(handboard|somni)\.([a-z0-9_]+)(?:\.(.+))?$"
)
_SIZE_PATTERN = re.compile(
    r"([eE+\-\.\d]+)\s*([kmgtpezyKMGTPEZY])?(i)?([bB])"
)
# 当前进程已挂文件 sink 的 (产品, 服务, 层)。不在其中的记录仍进公共日志，避免丢日志。
_routes: set[tuple[str, str, str]] = set()
# 拦截器在单次 RPC 期间放入控制层 logger，供 run_rpc_call 把异常栈写进对应控制层文件。
rpc_access_logger: ContextVar[Any] = ContextVar("rpc_access_logger", default=None)


def file_log_format(record: dict) -> str:
    """返回带颜色码的文件日志模板，供 tail -f 按级别着色。"""
    color = _FILE_LEVEL_COLOR.get(record["level"].name, _ANSI_GREEN)
    record["extra"]["ansi"] = color
    return "{extra[ansi]}" + FILE_LOG_FORMAT + "{exception}" + _ANSI_RESET + "\n"


def current_log_path(log_dir: Path, when: date | None = None) -> Path:
    """返回指定日期（默认今天）的公共日志路径：YYYY-MM-DD_ubur_log。"""
    day = when or date.today()
    return log_dir / f"{day:%Y-%m-%d}{LOG_FILE_SUFFIX}"


def service_log_path(
    log_dir: Path,
    product: str,
    service: str,
    layer: str,
    when: date | None = None,
) -> Path:
    """返回某服务某一层在指定日期的日志路径。"""
    day = when or date.today()
    return log_dir / product / service / f"{day:%Y-%m-%d}_{layer}"


def discover_rpc_services() -> list[tuple[str, str]]:
    """扫描带 Servicer 的 rpc.py，得到 (产品, 服务目录)。天气等纯映射模块不单开日志。"""
    server_root = Path(__file__).resolve().parents[1] / "server"
    found: list[tuple[str, str]] = []
    for product in ("handboard", "somni"):
        base = server_root / product
        if not base.is_dir():
            continue
        for rpc_file in sorted(base.glob("*/rpc.py")):
            text = rpc_file.read_text(encoding="utf-8")
            if "Servicer" not in text:
                continue
            found.append((product, rpc_file.parent.name))
    return found


def route_record(record: dict) -> tuple[str, str, str] | None:
    """判定一条日志该进哪个服务文件。无法归类时返回 None，由公共日志接收。"""
    extra = record["extra"]
    product = extra.get("log_product")
    service = extra.get("log_service")
    if product and service and _is_grpc_service(str(product), str(service)):
        return str(product), str(service), LAYER_LOG
    return _route_by_module(str(record["name"]))


def _route_by_module(name: str) -> tuple[str, str, str] | None:
    match = _MODULE_RE.match(name)
    if match is None:
        return None
    product, service, _rest = match.group(1), match.group(2), match.group(3)
    if not _is_grpc_service(product, service):
        return None
    return product, service, LAYER_LOG


def _is_grpc_service(product: str, service: str) -> bool:
    return (product, service, LAYER_LOG) in _routes


def _common_filter(record: dict) -> bool:
    return route_record(record) not in _routes


def _route_filter(product: str, service: str, layer: str) -> Callable[[dict], bool]:
    expected = (product, service, layer)

    def _filter(record: dict) -> bool:
        return route_record(record) == expected

    return _filter


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
    """初始化控制台、公共日志，以及各 gRPC 服务的控制层/服务层文件。返回公共日志路径。"""
    log_dir = settings.log_dir_path
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = current_log_path(log_dir)
    sink_pattern = str(log_dir / f"{{time:YYYY-MM-DD}}{LOG_FILE_SUFFIX}")
    max_bytes = parse_log_size(settings.log_rotation_size)
    found = discover_rpc_services()
    _routes.clear()
    for product, service in found:
        _routes.add((product, service, LAYER_LOG))

    logger.remove()
    logger.configure(extra={"request_id": DEFAULT_REQUEST_ID})
    level = settings.log_level.upper()
    logger.add(sys.stderr, level=level, format=CONSOLE_LOG_FORMAT)
    logger.add(
        sink_pattern,
        level=level,
        format=file_log_format,
        rotation=daily_or_size_rotation(max_bytes),
        retention=settings.log_retention,
        encoding="utf-8",
        enqueue=True,
        filter=_common_filter,
    )
    for product, service in found:
        service_dir = log_dir / product / service
        service_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(service_dir / "{time:YYYY-MM-DD}_log"),
            level=level,
            format=file_log_format,
            rotation=daily_or_size_rotation(max_bytes),
            retention=settings.log_retention,
            encoding="utf-8",
            enqueue=True,
            filter=_route_filter(product, service, LAYER_LOG),
        )
    logger.info(
        "日志已初始化，级别={}，公共文件={}，单文件上限={}，gRPC 服务数={}",
        level,
        log_file,
        settings.log_rotation_size,
        len(found),
    )
    return log_file
