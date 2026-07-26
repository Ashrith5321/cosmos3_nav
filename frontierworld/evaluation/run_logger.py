"""Experiment tracking.

CSV and JSONL are always available and are the source of truth. TensorBoard
and W&B are optional: enable them in the config only if the package is
installed, otherwise logging degrades to files with a warning rather than
taking the run down.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def make_run_id(name: str, config_hash: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{name}_{config_hash}"


class RunLogger:
    """Append-only metric log for one run."""

    def __init__(
        self,
        run_dir: str | Path,
        logging_cfg: Any,
        run_id: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = logging_cfg
        self.run_id = run_id

        self._jsonl = (
            (self.run_dir / "run.jsonl").open("a")
            if getattr(logging_cfg, "jsonl", True)
            else None
        )
        self._csv_path = self.run_dir / "run.csv"
        self._csv_file = None
        self._csv_writer: csv.DictWriter | None = None
        self._csv_columns: list[str] = []

        self._tb = self._init_tensorboard()
        self._wandb = self._init_wandb(config)

    # -- backends --------------------------------------------------------

    def _init_tensorboard(self):
        if not getattr(self.cfg, "tensorboard", False):
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning(
                "logging.tensorboard is true but tensorboard is not installed; "
                "falling back to CSV/JSONL only"
            )
            return None
        return SummaryWriter(log_dir=str(self.run_dir / "tb"))

    def _init_wandb(self, config: dict[str, Any] | None):
        if not getattr(self.cfg, "wandb", False):
            return None
        try:
            import wandb
        except ImportError:
            logger.warning(
                "logging.wandb is true but wandb is not installed; "
                "falling back to CSV/JSONL only"
            )
            return None
        wandb.init(
            project=getattr(self.cfg, "wandb_project", "frontierworld"),
            name=self.run_id,
            dir=str(self.run_dir),
            config=config or {},
        )
        return wandb

    # -- logging ---------------------------------------------------------

    def log(self, record: dict[str, Any], step: int | None = None) -> None:
        payload = {"run_id": self.run_id, **record}
        if step is not None:
            payload["step"] = step

        if self._jsonl is not None:
            self._jsonl.write(json.dumps(payload, default=str) + "\n")
            self._jsonl.flush()

        if getattr(self.cfg, "csv", True):
            self._write_csv(payload)

        if self._tb is not None:
            for key, value in record.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._tb.add_scalar(key, value, step or 0)

        if self._wandb is not None:
            self._wandb.log(record, step=step)

    def _write_csv(self, payload: dict[str, Any]) -> None:
        flat = {k: v for k, v in payload.items() if not isinstance(v, (dict, list))}
        # New columns can appear mid-run (e.g. a metric only present at the
        # end of an episode); rewrite the header rather than dropping them.
        new_columns = [c for c in flat if c not in self._csv_columns]
        if new_columns:
            self._csv_columns.extend(new_columns)
            existing_rows: list[dict[str, Any]] = []
            if self._csv_path.exists():
                with self._csv_path.open() as handle:
                    existing_rows = list(csv.DictReader(handle))
            if self._csv_file is not None:
                self._csv_file.close()
            self._csv_file = self._csv_path.open("w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_columns)
            self._csv_writer.writeheader()
            for row in existing_rows:
                self._csv_writer.writerow(row)
        assert self._csv_writer is not None
        self._csv_writer.writerow(flat)
        assert self._csv_file is not None
        self._csv_file.flush()

    def write_summary(self, summary: dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(
            json.dumps({"run_id": self.run_id, **summary}, indent=2, default=str)
        )

    def close(self) -> None:
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if self._tb is not None:
            self._tb.close()
            self._tb = None
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

def environment_provenance() -> dict[str, Any]:
    """Facts about the machine and code version, saved with every run."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    info["git_commit"] = _git("rev-parse", "HEAD")
    info["git_dirty"] = bool(_git("status", "--porcelain"))
    for module in ("numpy", "torch", "habitat", "habitat_sim"):
        try:
            info[f"{module}_version"] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - provenance must never break a run
            info[f"{module}_version"] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:  # noqa: BLE001
        pass
    return info


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parent.parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return None
