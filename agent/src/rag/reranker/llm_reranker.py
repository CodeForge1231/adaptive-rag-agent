from typing import List

from langchain_core.documents import Document

from .base import BaseReranker


class LLMReranker(BaseReranker):
    """
    Reranker implementation that leverages a Large Language Model to score document relevance.

    Attributes
    ----------
    rerank_chain : Any
        The LangChain runnable or chain responsible for scoring document relevance.
    top_k : int
        The maximum number of reranked documents to return.
    """
    def __init__(self, rerank_chain, top_k: int):
        self.rerank_chain = rerank_chain
        self.top_k = top_k

    async def arerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Asynchronously rerank documents using an LLM-based scoring mechanism.

        Parameters
        ----------
        query : str
            The user query to compare documents against.
        documents : List[Document]
            The initial list of documents retrieved from the vector store.

        Returns
        -------
        List[Document]
            The top k documents sorted by LLM relevance scores.
        """
        if not documents:
            return []

        formatted_docs = "\n".join(
            [f"ID: {i}\nContent: {doc.page_content[:500]}..." for i, doc in enumerate(documents)]
        )

        try:
            response = await self.rerank_chain.ainvoke(
                {"query": query, "documents": formatted_docs}
            )

            rankings = response.get("rankings", [])

            scores_map = {item["index"]: item["score"] for item in rankings}

            sorted_indices = sorted(scores_map.keys(), key=lambda x: scores_map[x], reverse=True)

            reranked_docs = [documents[i] for i in sorted_indices if i < len(documents)]
            return reranked_docs[: self.top_k]

        except Exception:
            return documents[: self.top_k]
