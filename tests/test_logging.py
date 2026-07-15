from datetime import date
from pathlib import Path

from app.core.config import Settings
from app.core.logging import LOG_FILE_SUFFIX, current_log_path, setup_logging


def test_current_log_path_uses_date_and_suffix(tmp_path: Path) -> None:
    day = date(2026, 7, 15)
    path = current_log_path(tmp_path, when=day)
    assert path == tmp_path / f"2026-07-15{LOG_FILE_SUFFIX}"
    assert path.name == "2026-07-15_ubur_log"


def test_setup_logging_creates_today_dated_file(tmp_path: Path) -> None:
    settings = Settings(log_dir=str(tmp_path), log_level="INFO")
    log_file = setup_logging(settings)
    assert log_file.exists()
    assert log_file.name == f"{date.today():%Y-%m-%d}_ubur_log"
    assert log_file.parent == tmp_path
