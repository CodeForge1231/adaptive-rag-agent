import os
from functools import lru_cache
from pathlib import Path


@lru_cache
def detect_project_root(anchors=("settings.yaml",)) -> Path:
    if "PROJECT_ROOT" in os.environ:
        return Path(os.environ["PROJECT_ROOT"]).resolve()

    # Walk up from CWD to allow running app from any subdirectory
    current = Path.cwd().resolve()

    for parent in (current, *current.parents):
        for anchor in anchors:
            if (parent / anchor).exists():
                return parent

    # Fail fast: project root is a hard requirement
    raise RuntimeError(f"Cannot detect project root. anchors={anchors}, cwd={current}")
