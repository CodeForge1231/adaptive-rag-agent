from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage


class BaseLLM(ABC):
    """
    Abstract base class for LLM implementations.
    Provides a unified interface for text generation.
    """

    @abstractmethod
    def generate(self, messages: list[BaseMessage]) -> str:
        """
        Generate a text response from a list of structured messages.

        Parameters
        ----------
        messages : list[BaseMessage]
            A sequence of messages (System, Human, AI) representing 
            the conversation history.

        Returns
        -------
        str
            The generated text content from the model.
        """
        pass

    @property
    @abstractmethod
    def runnable(self):
        """
        Return the underlying LangChain runnable instance.

        This property allows the LLM to be used within LangChain 
        chains and sequences.

        Returns
        -------
        Any
            The provider-specific LangChain chat model object.
        """
        pass
