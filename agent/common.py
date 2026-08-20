"""Shared helpers for the scheduled entrypoints — ingest, lint, and snapshot.

Not wiki_query.py: that one is manual and interactive, so it neither writes a
structured log nor has a launchd log to cap.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = _ROOT / "logs"

# The structured log records every tool call with its full JSON result, so a
# daily ingest writes megabytes a month and nothing ever removed them (27.8 MB
# by August 2026). 5 MB x 4 keeps roughly a quarter's history.
_LOG_MAX_BYTES = 5_000_000
_LOG_BACKUP_COUNT = 3

# The launchd job's raw stdout/stderr file, named in the .plist's
# StandardOutPath and passed back to the script via EnvironmentVariables.
LAUNCHD_LOG_ENV = "WIKI_LAUNCHD_LOG"
_LAUNCHD_LOG_MAX_BYTES = 5_000_000
_LAUNCHD_LOG_KEEP_BYTES = 1_000_000


def setup_logger(name: str) -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = RotatingFileHandler(
        log_path, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def trim_launchd_log(logger: logging.Logger = None) -> None:
    """Cap the launchd stdout/stderr log, keeping its tail.

    launchd owns that file: it opens it when the job starts and this process's
    stdout *is* that descriptor, so RotatingFileHandler can't manage it and
    renaming it would send the rest of the run's output to the renamed inode.
    Rewriting the tail in place keeps the inode, and because the descriptor is
    O_APPEND the run's later output lands after the retained tail.

    A no-op when WIKI_LAUNCHD_LOG is unset, which is every direct run.
    """
    path_str = os.getenv(LAUNCHD_LOG_ENV)
    if not path_str:
        return

    path = Path(path_str)
    try:
        size = path.stat().st_size
        if size <= _LAUNCHD_LOG_MAX_BYTES:
            return
        with open(path, "r+b") as f:
            f.seek(size - _LAUNCHD_LOG_KEEP_BYTES)
            f.readline()  # drop the partial line the offset landed inside
            tail = f.read()
            f.seek(0)
            f.write(tail)
            f.truncate()
    except OSError as e:
        if logger:
            logger.warning(f"Could not trim launchd log {path}: {e}")
        return

    if logger:
        logger.info(
            f"Trimmed {path.name} from {size / 1e6:.1f} MB to its last "
            f"{len(tail) / 1e6:.1f} MB"
        )
