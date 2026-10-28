from datetime import date
from pathlib import Path

from loguru import logger

from app.core.config import Settings
from app.core.logging import (
    LAYER_CONTROLLER,
    LAYER_SERVICE,
    LOG_FILE_SUFFIX,
    current_log_path,
    discover_rpc_services,
    parse_log_size,
    service_log_path,
    setup_logging,
)


def test_current_log_path_uses_date_and_suffix(tmp_path: Path) -> None:
    day = date(2026, 7, 15)
    path = current_log_path(tmp_path, when=day)
    assert path == tmp_path / f"2026-07-15{LOG_FILE_SUFFIX}"
    assert path.name == "2026-07-15_ubur_log"


def test_parse_log_size_matches_common_units() -> None:
    assert parse_log_size("100 MB") == 100_000_000
    assert parse_log_size("1 KiB") == 1024
    assert parse_log_size("512 B") == 512


def test_setup_logging_creates_today_dated_file(tmp_path: Path) -> None:
    settings = Settings(log_dir=str(tmp_path), log_level="INFO")
    log_file = setup_logging(settings)
    assert log_file.exists()
    assert log_file.name == f"{date.today():%Y-%m-%d}_ubur_log"
    assert log_file.parent == tmp_path


def test_file_log_includes_request_id_from_context(tmp_path: Path) -> None:
    settings = Settings(log_dir=str(tmp_path), log_level="INFO")
    log_file = setup_logging(settings)
    with logger.contextualize(request_id="rid-abc123"):
        logger.info("带 request_id 的探测日志")
    # enqueue=True 时需等异步写入完成
    logger.complete()
    content = log_file.read_text(encoding="utf-8")
    assert "rid-abc123" in content
    assert "带 request_id 的探测日志" in content


def test_file_log_colors_level_for_tail(tmp_path: Path) -> None:
    settings = Settings(log_dir=str(tmp_path), log_level="DEBUG")
    log_file = setup_logging(settings)
    logger.info("正常探测")
    logger.warning("警告探测")
    logger.error("错误探测")
    logger.complete()
    content = log_file.read_text(encoding="utf-8")
    assert "\033[32m" in content.split("正常探测")[0]
    warning = content.split("警告探测")[0].rsplit("\033[", 1)[-1]
    error = content.split("错误探测")[0].rsplit("\033[", 1)[-1]
    assert warning.startswith("33m")
    assert error.startswith("31m")


def test_file_log_uses_default_request_id_when_absent(tmp_path: Path) -> None:
    settings = Settings(log_dir=str(tmp_path), log_level="INFO")
    log_file = setup_logging(settings)
    logger.info("无上下文的探测日志")
    logger.complete()
    content = log_file.read_text(encoding="utf-8")
    assert "无上下文的探测日志" in content
    # 缺省占位，避免 extra[request_id] KeyError
    assert " | - | " in content or content.count("| - |") >= 1


def test_size_rotation_archives_full_file_and_continues_same_day(
    tmp_path: Path,
) -> None:
    settings = Settings(
        log_dir=str(tmp_path),
        log_level="INFO",
        log_rotation_size="300 B",
    )
    active = setup_logging(settings)
    for index in range(40):
        logger.info("size-rotation-probe-{}", index)
    logger.complete()

    day_prefix = f"{date.today():%Y-%m-%d}{LOG_FILE_SUFFIX}"
    archived = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(day_prefix) and path.name != day_prefix
    ]
    assert archived, "超大小后应归档至少一个带时间戳后缀的当日文件"
    assert active.exists()
    assert active.name == day_prefix
    assert "size-rotation-probe" in active.read_text(encoding="utf-8")


def test_ask_logs_use_one_file(tmp_path: Path) -> None:
    settings = Settings(log_dir=str(tmp_path), log_level="INFO")
    setup_logging(settings)
    ask_logger = logger.patch(
        lambda record: record.update(name="app.server.somni.ask.rpc")
    )
    ask_logger.info("Ask 单文件探测")
    logger.complete()

    day = date.today()
    single = tmp_path / "somni" / "ask" / f"{day:%Y-%m-%d}_log"
    assert single.exists()
    assert "Ask 单文件探测" in single.read_text(encoding="utf-8")
    assert not (tmp_path / "somni" / "ask" / f"{day:%Y-%m-%d}_service").exists()
    assert not (tmp_path / "somni" / "ask" / f"{day:%Y-%m-%d}_controller").exists()


def test_discover_rpc_services_skips_mapper_only_packages() -> None:
    found = set(discover_rpc_services())
    assert ("somni", "audio") in found
    assert ("somni", "facts") in found
    assert ("somni", "applewatch") in found
    assert ("handboard", "audio") in found
    assert ("somni", "weather") not in found


def test_service_and_access_logs_share_one_file(tmp_path: Path) -> None:
    settings = Settings(log_dir=str(tmp_path), log_level="INFO")
    common = setup_logging(settings)
    service_logger = logger.patch(
        lambda record: record.update(name="app.server.somni.audio.catalog")
    )
    controller_logger = logger.bind(
        log_product="somni",
        log_service="audio",
        log_layer=LAYER_CONTROLLER,
    )
    service_logger.warning("目录服务探测")
    controller_logger.info("控制层探测")
    logger.info("公共日志探测")
    logger.complete()

    day = date.today()
    single = tmp_path / "somni" / "audio" / f"{day:%Y-%m-%d}_log"
    text = single.read_text(encoding="utf-8")
    common_text = common.read_text(encoding="utf-8")

    assert "目录服务探测" in text
    assert "控制层探测" in text
    assert "公共日志探测" in common_text
    assert "目录服务探测" not in common_text
    assert not service_log_path(tmp_path, "somni", "audio", LAYER_SERVICE).exists()
    assert not service_log_path(tmp_path, "somni", "audio", LAYER_CONTROLLER).exists()
