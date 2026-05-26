import logging

from rich.logging import RichHandler

from .config import LoggingConfig


def setup_logging(cfg: LoggingConfig) -> None:
    """
    Configure the global logging system for the application.

    Parameters
    ----------
    cfg : LoggingConfig
        Configuration object containing logging levels and display options.

    Returns
    -------
    None
    """
    # Globally disable all logging if specified in the configuration
    if not cfg.enabled:
        logging.disable(logging.CRITICAL)
        return

    # Map configuration string level to standard logging constants
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    # Configure root logger with Rich handler for beautiful terminal output
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(
                rich_tracebacks=cfg.rich_tracebacks,
                show_time=cfg.show_time,
                show_level=cfg.show_level,
                show_path=cfg.show_path,
            )
        ],
    )
