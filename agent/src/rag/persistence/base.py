import logging

logger = logging.getLogger(__name__)


class PersistenceContext:
    """
    Runtime context for managing database resources and repository lifecycles.
    """

    def __init__(self, engine, repositories):
        """
        Initialize the persistence context with active infrastructure components.

        Parameters
        ----------
        engine : sqlalchemy.ext.asyncio.AsyncEngine
            An initialized SQLAlchemy asynchronous engine.
        repositories : Any
            The repository registry or collection associated with this context.
        """
        self.engine = engine
        self.repositories = repositories
        logger.info("PersistenceContext initialized")

    async def shutdown(self):
        """
        Gracefully shut down the database engine and release resources.
        """
        await self.engine.dispose()
        logger.info("Persistence engine disposed")
