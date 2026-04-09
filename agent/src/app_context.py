import asyncio
from src.rag.embeddings.base import BaseEmbeddingModel

class AppContext:
    def __init__(
        self,
        *,
        settings: dict,
        embeddings: BaseEmbeddingModel,
    ):
        self.settings = settings
        self.embeddings = embeddings

    async def shutdown(self):
        if self.persistence is not None:
            await self.persistence.shutdown()
        loop = asyncio.get_running_loop()
        await loop.shutdown_default_executor()
