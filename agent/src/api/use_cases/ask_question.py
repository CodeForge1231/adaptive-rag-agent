from langchain_core.messages import AIMessage, HumanMessage


class AskQuestionUseCase:
    """
    Use case for processing user questions through a RAG orchestrator.
    """

    def __init__(self, orchestrator):
        """
        Initialize with a RAG orchestrator instance.
        """
        self.orchestrator = orchestrator

    async def execute(self, data):
        """
        Execute the RAG pipeline for a given user question.

        Parameters
        ----------
        data : AskRequest
            Input data containing the question, history, and user info.

        Returns
        -------
        dict
            A dictionary containing the answer, updated history, and sources.
        """

        state = {
            "question": data.question,
            "messages": self._ui_to_langchain(data.messages),
            "documenst": [],
            "answer": "",
            "user": {"user_id": data.user.user_id},
        }

        result = await self.orchestrator.run(state)

        print(f"result: {result}")
        return {
            "answer": result["answer"],
            "messages": self._langchain_to_ui(result["messages"]),
            "documents": result.get("documents") or [],
        }

    def _ui_to_langchain(self, messages):
        """
        Convert UI message schemas to LangChain message objects.

        Parameters
        ----------
        messages : list of MessageSchema
            A list of messages from the frontend, where each message has 
            'role' and 'content' attributes.

        Returns
        -------
        list of Union[HumanMessage, AIMessage]
            A list of LangChain-compatible message objects.
        """
        result = []

        for msg in messages:
            if msg.role == "user":
                result.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                result.append(AIMessage(content=msg.content))

        return result

    def _langchain_to_ui(self, messages):
        """
        Convert LangChain message objects back to UI-friendly dictionaries.

        Parameters
        ----------
        messages : list of BaseMessage
            A list of LangChain objects (HumanMessage, AIMessage, etc.).

        Returns
        -------
        list of dict
            A list of dictionaries containing 'role' and 'content' keys, 
            ready for serialization to the frontend.
        """
        result = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})

            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})

        return result
