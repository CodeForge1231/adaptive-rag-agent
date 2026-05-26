import os

import yaml
from dotenv import load_dotenv

from src.core.paths import detect_project_root


def load_config() -> dict:
    # Resolve project root once to make config loading location-independent
    BASE_DIR = detect_project_root()
    os.environ.setdefault("PROJECT_ROOT", str(BASE_DIR))

    # Load environment variables early to allow secrets in config
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Read raw YAML and expand env vars before parsing
    raw = (BASE_DIR / "settings.yaml").read_text()
    raw = os.path.expandvars(raw)

    return yaml.safe_load(raw)
