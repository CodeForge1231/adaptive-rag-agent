from langchain_core.documents import Document


class AddDocumentUseCase:
    def __init__(self, vectorstore):
        """
        Initialize the use case with a vector store instance.
        """
        self.vectorstore = vectorstore

    async def execute(self, data) -> dict:
        """
        Convert input data to a Document and persist it.

        Parameters
        ----------
        data : AddDocumentRequest
            The data transfer object containing text, id, and metadata.

        Returns
        -------
        dict
            A status dictionary indicating the operation outcome.
        """
        # Wrap raw text and metadata into a LangChain Document object
        document = Document(
            page_content=data.text,
            metadata=data.metadata,
        )

        # Asynchronously add the document to the vector database
        await self.vectorstore.aadd_documents(
            documents=[document],
            ids=[data.id],
        )

        return {
            "success": True
        }