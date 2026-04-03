from rich.traceback import install

from .config import TracebackConfig


def setup_traceback(cfg: TracebackConfig) -> None:
    # Allow disabling enhanced tracebacks completely
    if not cfg.enabled:
        return

    install(
        show_locals=cfg.show_locals,
        suppress=cfg.suppress or [],
    )
