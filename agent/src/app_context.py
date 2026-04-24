import asyncio
from src.rag.persistence.sqlalchemy.base import DocumentHistoryRepository, UserProfileRepository
from src.rag.persistence.base import PersistenceContext
from src.rag.retriever.base import BaseRetriever
from src.rag.embeddings.base import BaseEmbeddingModel
from src.rag.vectorstore.base import BaseVectorStore

class AppContext:
    def __init__(
        self,
        *,
        settings: dict,
        embeddings: BaseEmbeddingModel,
        vectorstore: BaseVectorStore,
        retriever: BaseRetriever,
        persistence: PersistenceContext,
        user_profiles: UserProfileRepository,
        document_history: DocumentHistoryRepository,
    ):
        self.settings = settings
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.persistence = persistence
        self.user_profiles = user_profiles
        self.document_history = document_history

    async def shutdown(self):
        if self.persistence is not None:
            await self.persistence.shutdown()
        loop = asyncio.get_running_loop()
        await loop.shutdown_default_executor()
