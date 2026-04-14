import asyncio
from src.rag.embeddings.base import BaseEmbeddingModel
from src.rag.vectorstore.base import BaseVectorStore

class AppContext:
    def __init__(
        self,
        *,
        settings: dict,
        embeddings: BaseEmbeddingModel,
        vectorstore: BaseVectorStore,
    ):
        self.settings = settings
        self.embeddings = embeddings
        self.vectorstore = vectorstore
    async def shutdown(self):
        if self.persistence is not None:
            await self.persistence.shutdown()
        loop = asyncio.get_running_loop()
        await loop.shutdown_default_executor()
