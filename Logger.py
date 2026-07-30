"""logger.py

Centralized logging utilities for the Safe RL AMR project.

Provides two complementary facilities used across ``train.py``,
``evaluate.py``, and any future scripts:

    1. :func:`get_logger` -- a standard Python ``logging.Logger`` with a
       colored console handler and an optional rotating file handler, for
       human-readable run/status/error messages (replacing ad-hoc
       ``print`` calls).
    2. :class:`MetricLogger` -- a thin wrapper around
       ``torch.utils.tensorboard.SummaryWriter`` that additionally mirrors
       every scalar to a flat CSV file, so training curves remain
       inspectable (e.g. via :func:`visualizer.plot_training_curves`'s CSV
       fallback path, or plain ``pandas.read_csv``) even in environments
       where loading TensorBoard event files is inconvenient.

Neither facility is required by the rest of the codebase -- ``train.py``
and ``evaluate.py`` currently log via plain ``print`` and their own
``SummaryWriter`` -- but both are drop-in compatible with that existing
code and are the recommended entry point for any new scripts.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, TextIO, Union

from torch.utils.tensorboard import SummaryWriter

# --------------------------------------------------------------------------- #
# Console color codes (no external dependency)
# --------------------------------------------------------------------------- #

_RESET = "\x1b[0m"
_COLORS: Dict[int, str] = {
    logging.DEBUG: "\x1b[38;5;244m",   # grey
    logging.INFO: "\x1b[38;5;39m",     # blue
    logging.WARNING: "\x1b[38;5;214m",  # orange
    logging.ERROR: "\x1b[38;5;196m",   # red
    logging.CRITICAL: "\x1b[1;38;5;196m",  # bold red
}


class _ColorFormatter(logging.Formatter):
    """A ``logging.Formatter`` that colorizes the level name for TTY output."""

    def __init__(self, use_color: bool) -> None:
        """Initializes the formatter.

        Args:
            use_color: Whether to wrap the level name in ANSI color codes
                (disabled automatically for non-TTY streams, e.g. when
                output is redirected to a file).
        """
        super().__init__(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record, colorizing the level name when enabled.

        Args:
            record: The log record to format.

        Returns:
            The formatted log line.
        """
        if self.use_color:
            color = _COLORS.get(record.levelno, "")
            record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


# --------------------------------------------------------------------------- #
# Standard logger
# --------------------------------------------------------------------------- #

def get_logger(
    name: str = "safe_rl_amr",
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console_stream: TextIO = sys.stdout,
) -> logging.Logger:
    """Builds (or retrieves) a configured logger with console + file handlers.

    Safe to call repeatedly with the same ``name`` (e.g. once per module):
    handlers are only attached the first time a given logger name is
    configured, so re-invoking this function does not duplicate log lines.

    Args:
        name: Logger name (also used as the log-file stem when ``log_dir``
            is given).
        log_dir: If provided, a ``{name}.log`` file is created under this
            directory and every message is additionally written there
            (uncolored, since it's a plain text file). The directory is
            created if it does not exist.
        level: Minimum severity level to emit.
        console_stream: Stream the console handler writes to (defaults to
            ``sys.stdout``).

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        # Already configured (e.g. called earlier in the same process from
        # another module) -- avoid attaching duplicate handlers.
        return logger

    is_tty = hasattr(console_stream, "isatty") and console_stream.isatty()
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setLevel(level)
    console_handler.setFormatter(_ColorFormatter(use_color=is_tty))
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir_path / f"{name}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(_ColorFormatter(use_color=False))
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Metric logger (TensorBoard + CSV)
# --------------------------------------------------------------------------- #

class MetricLogger:
    """Unified scalar-metric logger writing to both TensorBoard and a CSV file.

    Every call to :meth:`log_scalar` / :meth:`log_dict` appends to the
    ``SummaryWriter`` (for live TensorBoard viewing) and to a flat
    ``metrics.csv`` file with columns ``step, tag, value, wall_time`` (for
    simple ``pandas``/``csv`` post-processing without needing the
    ``tensorboard`` package installed to *read* logs written on another
    machine).

    Example:
        >>> metric_logger = MetricLogger(log_dir="runs/safe_sac_001")
        >>> metric_logger.log_dict({"train/critic_loss": 0.42}, step=1000)
        >>> metric_logger.close()
    """

    def __init__(self, log_dir: Union[str, Path]) -> None:
        """Initializes the TensorBoard writer and CSV file.

        Args:
            log_dir: Directory to write TensorBoard event files and
                ``metrics.csv`` into. Created if it does not exist.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self._csv_path = self.log_dir / "metrics.csv"
        csv_is_new = not self._csv_path.exists()
        self._csv_file = open(self._csv_path, mode="a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if csv_is_new:
            self._csv_writer.writerow(["step", "tag", "value", "wall_time"])

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Logs a single scalar to both TensorBoard and the CSV file.

        Args:
            tag: Metric name (e.g. ``"train/critic_loss"``).
            value: Scalar value.
            step: Global step (e.g. environment step or gradient update
                count) this value corresponds to.
        """
        self.writer.add_scalar(tag, value, step)
        self._csv_writer.writerow([step, tag, float(value), time.time()])

    def log_dict(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """Logs every entry of a flat metrics dict.

        Args:
            metrics: Mapping of metric name to scalar value (e.g. the dict
                returned by ``SafeSACAgent.update`` or
                ``evaluate_agent``).
            step: Global step these values correspond to.
            prefix: Optional prefix prepended to every tag (with a ``/``
                separator), e.g. ``prefix="train"`` turns
                ``"critic_loss"`` into ``"train/critic_loss"``.
        """
        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.log_scalar(tag, float(value), step)

    def flush(self) -> None:
        """Flushes both the TensorBoard writer and the CSV file to disk."""
        self.writer.flush()
        self._csv_file.flush()

    def close(self) -> None:
        """Closes the TensorBoard writer and the CSV file."""
        self.writer.close()
        self._csv_file.close()

    def __enter__(self) -> "MetricLogger":
        """Enables use as a context manager: ``with MetricLogger(...) as m:``."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Ensures :meth:`close` runs when leaving a ``with`` block."""
        self.close()