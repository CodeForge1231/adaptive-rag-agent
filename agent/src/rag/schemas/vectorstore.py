from typing import Any, List, TypedDict

import numpy as np


class VectorStoreData(TypedDict):
    """
    A structured representation of the internal state and content of a vector store.
    """

    ids: List[str]
    """
    A list of unique identifiers for each stored document or entry.
    """

    embeddings: np.ndarray
    """
    A high-dimensional NumPy array representing the vectorized numerical 
    embeddings for all stored documents.
    """

    documents: List[str]
    """
    The raw text content or serialized document strings corresponding to each embedding.
    """

    metadatas: List[dict[str, Any]]
    """
    A collection of dictionaries containing additional attributes, filters, 
    or labels associated with each document.
    """