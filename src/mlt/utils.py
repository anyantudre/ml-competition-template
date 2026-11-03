"""Small helpers shared by every pipeline: logging, seeding, hashing, json and timing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

LOG = logging.getLogger("mlt")


def setup_logging(level: str = "INFO") -> None:
    if LOG.handlers:  # already configured; keep the current level (tune lowers it)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    LOG.addHandler(handler)
    LOG.setLevel(level)
    LOG.propagate = False


def seed_everything(seed: int) -> None:
    """Seed python, numpy and, when it is already imported, torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def file_hash(path: str | Path, n: int = 10) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def array_hash(values, n: int = 10) -> str:
    return hashlib.md5(np.ascontiguousarray(values).tobytes()).hexdigest()[:n]


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        )
        if out.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def write_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


@contextmanager
def timer(label: str):
    start = time.perf_counter()
    yield
    LOG.info("%s took %.1fs", label, time.perf_counter() - start)


def get_device(preference: str = "auto"):
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require(module: str, extra: str):
    """Import an optional dependency or explain which extra installs it."""
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - message only
        raise ImportError(
            f"'{module}' is needed for this step. Install it with: uv pip install -e '.[{extra}]'"
        ) from exc
