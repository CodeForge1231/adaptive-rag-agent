from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.rag.persistence.config import (
    BaseDatabaseConfig,
    PostgresDatabaseConfig,
    SQLiteDatabaseConfig,
)


def create_session(db_cfg: BaseDatabaseConfig):
    """
    Initialize an asynchronous database engine and session factory.

    Parameters
    ----------
    db_cfg : BaseDatabaseConfig
        The configuration object containing connection parameters.

    Returns
    -------
    tuple[AsyncEngine, async_sessionmaker]
        A tuple containing the initialized SQLAlchemy asynchronous engine 
        and the configured session factory.

    Raises
    ------
    TypeError
        If the provided configuration object does not match any supported 
        database types.
    """
    match db_cfg:
        case SQLiteDatabaseConfig() as cfg:
            # Ensure the local directory for the database file exists
            db_dir = Path(cfg.path)
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / f"{cfg.database}.db"

            # Construct aiosqlite connection string
            url = f"sqlite+aiosqlite:///{db_path.resolve()}"

        case PostgresDatabaseConfig() as cfg:
            # Construct asyncpg connection string with credentials
            url = (
                f"postgresql+asyncpg://"
                f"{cfg.user}:{cfg.password}"
                f"@{cfg.host}:{cfg.port}/{cfg.database}"
            )

        case _:
            raise TypeError(f"Unsupported database config type: {type(db_cfg)}")

    # Initialize the asynchronous engine with the resolved URL
    engine = create_async_engine(url)

    # Configure the session maker for non-blocking database interactions
    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )

    return engine, session_factory
