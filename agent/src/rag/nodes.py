import json
import logging

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.core.observability import observability
from src.rag.schemas.graph_state import RAGState

logger = logging.getLogger(__name__)


class RAGNodes:
    """
    Collection of node functions for the Agentic RAG graph.
    """
    def __init__(self, config, chains, retriever, reranker, profile_repo, document_history_repo):
        self.chains = chains
        self.retriever = retriever
        self.state_schema = RAGState
        self.reranker = reranker
        self.history_settings = config.get("rag", {}).get("memory", {})
        self.profile_repo = profile_repo
        self.document_history_repo = document_history_repo

    def _get_history(self, messages, key):
        """
        Retrieve a window of messages from the conversation history based on a size key.

        Args:
            messages: Full list of conversation messages.
            key (str): History window identifier ('small', 'medium', 'large').

        Returns:
            List: Filtered and truncated list of messages.
        """
        if not messages:
            return []

        cfg = self.history_settings.get(key, {})
        window_size = cfg.get("window_size", 6)
        human_only = cfg.get("human_only", False)

        if human_only:
            messages = [m for m in messages if isinstance(m, HumanMessage)]

        return messages[-window_size:]

    def _format_chat_history(self, messages):
        """Format a list of messages into a single string for prompt injection."""
        return "\n".join(f"{m.type}: {m.content}" for m in messages)

    @observability.time_it_async("Add User Message to Messages")
    async def add_user_message(self, state: RAGState):
        """Append the current raw question to the message history if not already present."""
        question = state.get("question")

        if not question:
            return {}

        messages = state.get("messages", [])

        if messages and isinstance(messages[-1], HumanMessage):
            return {}

        return {"messages": messages + [HumanMessage(content=question)]}

    @observability.time_it_async("Routing Request")
    async def route_request(self, state: RAGState):
        """Determine the next step in the graph based on user intent and history."""
        logger.info("Routing request with history")
        messages = self._get_history(state.get("messages"), "medium")
        chat_history = "\n".join(f"{m.type}: {m.content}" for m in messages)
        result = await self.chains.router.ainvoke(
            {"question": state["question"], "chat_history": chat_history}
        )

        return {"route": result.datasource}

    @observability.time_it_async("Clarification Check")
    async def check_clarification(self, state: RAGState):
        """Evaluate if enough context exists to perform a high-quality retrieval."""
        ctx = state.get("resolved_context", {})
        profile = state.get("user", {}).get("profile", {})

        if not ctx.get("topic") and not profile.get("topics"):
            return {"clarified": False, "missing_info": ["topic"]}

        return {"clarified": True, "missing_info": []}

    @observability.time_it_async("Query Rewriting")
    async def rewrite_query(self, state: RAGState):
        """Transform the user question into a search-optimized query with metadata."""
        logger.info("Rewriting query for retrieval")

        messages = self._get_history(state.get("messages"), "small")
        context = "\n".join(f"{m.type}: {m.content}" for m in messages)

        intent = await self.chains.rewriter.ainvoke({"context": context})
        logger.info(
            "Retriever intent extracted",
            extra={"retriever_query": intent.retriever_query, "metadata": intent.metadata.dict()},
        )

        return {
            "rewritten_query": intent.retriever_query,
            "retriever_metadata": intent.metadata.dict(),
        }

    @observability.time_it_async("Vector Retrieval")
    async def retrieve_docs(self, state: RAGState):
        """Fetch relevant documents from the vector store using filters and semantic search."""
        logger.info("Retrieving documents")

        query = state.get("rewritten_query") or state["question"]

        user = state.get("user", {})
        profile = user.get("profile", {}) or {}

        metadata = state.get("retriever_metadata", {}) or {}

        strict_filters = {}

        logger.warning(f"profile: {profile}")

        # base metadata filters
        for k, v in metadata.items():
            if v is not None:
                strict_filters[k] = v

        # profile filters
        if profile.get("age_group"):
            strict_filters["age_group"] = profile["age_group"]

        if profile.get("level"):
            strict_filters["level"] = profile["level"]

        logger.info(f"STRICT filters: {strict_filters}")

        docs = await self.retriever.aretrieve(
            query=query,
            filters=strict_filters if strict_filters else None,
        )

        if not docs:
            logger.warning("No docs even without filters → full retrieval")

            docs = await self.retriever.aretrieve(
                query=query,
                filters=None,
            )

        logger.info(
            "Documents retrieved",
            extra={
                "query": query,
                "docs_count": len(docs),
                "used_strict": bool(strict_filters),
            },
        )

        logger.warning(f"docs: {docs}")
        return {"documents": docs}

    @observability.time_it_async("Document Reranking")
    async def rerank_docs(self, state: RAGState):
        """Re-order retrieved documents based on cross-encoder scoring for relevance."""
        logger.info("Reranking documents")

        docs = state.get("documents", [])
        query = state.get("rewritten_query") or state["question"]

        if not docs:
            return {"documents": []}

        reranked_docs = await self.reranker.arerank(query=query, documents=docs)

        logger.info(
            "Documents reranked",
            extra={"original_count": len(docs), "final_count": len(reranked_docs)},
        )

        return {"documents": reranked_docs}

    @observability.time_it_async("Asking Clarifying Question")
    async def ask_clarifying_question(self, state: RAGState):
        """Generate a clarifying question to resolve missing parameters in the query."""
        logger.info("Asking clarifying question")

        messages = self._get_history(state.get("messages"), "medium")
        # chat_history = "\n".join(f"{m.type}: {m.content}" for m in messages)
        missing = state.get("missing_info", [])

        answer = await self.chains.clarification_asker.ainvoke(
            {"missing_info": missing, "messages": messages}
        )

        return {"answer": answer}

    @observability.time_it_async("Simple LLM Response")
    async def simple_response(self, state: RAGState):
        """Provide a direct LLM response for queries that do not require document retrieval."""
        logger.info("Generating simple response")

        messages = self._get_history(state.get("messages"), "large")
        answer = await self.chains.simple_generator.ainvoke({"messages": messages})

        return {"answer": answer}

    @observability.time_it_async("Final answer with RAG")
    async def generate(self, state: RAGState):
        """Synthesize a final response using the reranked documents and conversation history."""
        logger.info("Generating answer")

        messages = self._get_history(state.get("messages"), "large")
        documents = state.get("documents", [])

        context_parts = []
        sources_set = set()

        # Format retrieved chunks for the generation prompt
        for i, doc in enumerate(documents, start=1):
            if not isinstance(doc, Document):
                continue

            content = doc.page_content.strip()
            if not content:
                continue

            title = doc.metadata.get("title", "Untitled Document").strip()
            source = doc.metadata.get("source", "Unknown Source").strip()

            sources_set.add(source)

            doc_block = f"DOCUMENT {i}: {title}\nCONTENT: {content}"
            context_parts.append(doc_block)

        context = "\n\n" + "-" * 30 + "\n\n".join(context_parts)

        answer = await self.chains.generate.ainvoke({"messages": messages, "context": context})

        return {
            "answer": answer,
            "messages": messages,
            "sources": sources_set,
        }

    @observability.time_it_async("Add AI Message to Messages")
    async def add_assistant_message(self, state: RAGState):
        """Add the assistant's final response to the conversation state."""
        answer = state.get("answer")
        if not answer:
            return {}

        messages = state.get("messages", [])

        if messages and isinstance(messages[-1], AIMessage):
            return {}

        return {"messages": messages + [AIMessage(content=answer)]}

    @observability.time_it_async("Resolve Context")
    async def resolve_context(self, state: RAGState):
        """Merge search intent metadata with persistent user profile attributes."""
        logger.info("Merging intent with user profile")

        intent = state.get("retriever_metadata", {})
        profile = state.get("user", {}).get("profile", {})

        resolved = {
            "topic": (
                intent.get("topic")
                or profile.get("topics", [])
            ),
            "level": (
                intent.get("level")
                or profile.get("level")
            ),
            "age_group": (
                intent.get("age_group")
                or profile.get("age_group")
            ),
        }

        logger.info(f"Resolved context: {resolved}")

        return {"resolved_context": resolved}

    @observability.time_it_async("Update User Profile")
    async def update_profile(self, state: RAGState):
        """Update and save the user's persistent profile based on new intent data."""
        user_id = state.get("user", {}).get("user_id")

        if not user_id:
            return {}

        profile = state.get("user", {}).get("profile") or {}
        intent = state.get("retriever_metadata", {})

        updated_profile = profile.copy()

        # level
        if intent.get("level"):
            updated_profile["level"] = intent["level"]

        # age_group
        if intent.get("age_group"):
            updated_profile["age_group"] = intent["age_group"]

        # topics
        if intent.get("topic"):
            updated_profile.setdefault("topics", [])

            topics = intent["topic"]

            # normalize to list
            if isinstance(topics, str):
                topics = [topics]

            for topic in topics:
                if topic not in updated_profile["topics"]:
                    updated_profile["topics"].append(topic)

        await self.profile_repo.save(user_id, updated_profile)

        return {"user_profile": updated_profile}

    @observability.time_it_async("Load User Profile")
    async def load_user_profile(self, state):
        """Fetch user data from the profile repository into the graph state."""
        if self.profile_repo is None:
            return state

        user_id = state["user"]["user_id"]
        profile = await self.profile_repo.load(user_id)

        state["user"]["profile"] = profile
        return state

    @observability.time_it_async("Load Document History")
    async def load_document_history(self, state: RAGState):
        """Retrieve a list of documents previously presented to the user."""
        if not self.document_history_repo:
            return {}

        user_id = state["user"]["user_id"]
        history = await self.document_history_repo.get_user_history(user_id)

        logger.warning(f"history: {history}")
        return {"documents_history": history}

    @observability.time_it_async("Log Sent Documents")
    async def log_documents(self, state: RAGState):
        """Record documents used in the final answer to prevent future redundancy."""
        logger.info("Logging used documents to history")

        user_id = state.get("user", {}).get("user_id")
        documents = state.get("documents", [])

        if not user_id or not documents:
            logger.warning("Skipping document logging: user_id or documents missing")
            return {}

        docs_to_log = []
        for doc in documents:
            if not isinstance(doc, Document):
                continue

            doc_id = doc.metadata.get("id")
            title = doc.metadata.get("title") or "Untitled Document"

            if doc_id:
                docs_to_log.append({"document_id": str(doc_id), "title": str(title)})

        if docs_to_log:
            try:
                await self.document_history_repo.log_sent_documents(
                    user_id=user_id, documents=docs_to_log
                )
                logger.info(f"Successfully logged documents for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to log documents: {e}", exc_info=True)

        return {}

    @observability.time_it_async("Detect Negative Feedback")
    async def detect_negative_feedback(self, state: RAGState):
        """Analyze user response to determine if retrieved information was unsatisfactory."""
        logger.info("Detecting feedback on retrieval")

        messages = self._get_history(state.get("messages"), "small")
        if not messages:
            return {"retrieval_feedback": "neutral"}

        last_message = messages[-1].content
        # We need a bit of history to understand context
        chat_history = "\n".join(f"{m.type}: {m.content}" for m in messages[:-1])

        result = await self.chains.feedback_detector.ainvoke(
            {"chat_history": chat_history, "last_message": last_message}
        )

        feedback_status = result.feedback if hasattr(result, "feedback") else "neutral"

        logger.info(f"Feedback detected: {feedback_status}")

        return {"retrieval_feedback": feedback_status}

    @observability.time_it_async("Feedback Decision")
    async def feedback_decision(self, state: RAGState):
        """Choose a correction path (rewrite or clarify) after negative feedback detection."""
        logger.info("Deciding next action based on negative feedback")

        messages = self._get_history(state.get("messages"), "small")
        chat_history = self._format_chat_history(messages)

        result = await self.chains.feedback_decision_maker.ainvoke(
            {
                "chat_history": chat_history,
            }
        )

        action = result.action
        logger.info(f"Feedback decision: {action}")

        return {"feedback_action": action}

    @observability.time_it_async("Ask Feedback Clarification")
    async def ask_feedback_clarify(self, state: RAGState):
        """Generate a response asking for details when retrieval results are rejected."""
        logger.info("Asking for feedback clarification")

        messages = self._get_history(state.get("messages"), "small")
        chat_history = self._format_chat_history(messages)

        answer = await self.chains.feedback_clarifier.ainvoke(
            {
                "chat_history": chat_history,
            }
        )

        return {"answer": answer}

    @observability.time_it_async("Rewrite Query with documents history")
    async def rewrite_with_history(self, state: RAGState):
        """Perform query rewriting while explicitly excluding previously seen documents."""
        logger.info("Rewriting query based on feedback")

        messages = self._get_history(state.get("messages"), "medium")
        chat_history = self._format_chat_history(messages)

        previous_query = state.get("rewritten_query") or state.get("question")

        history = state.get("documents_history", [])

        seen_titles = [entry["title"] for entry in history if entry.get("title")]

        new_query = await self.chains.history_rewriter.ainvoke(
            {
                "previous_query": previous_query,
                "seen_titles": "\n".join(seen_titles),
                "chat_history": chat_history,
            }
        )

        logger.info(f"Rewritten query: {new_query}")

        return {"rewritten_query": new_query}

    @observability.time_it_async("Retrieval with Exclusion")
    async def retrieve_with_exclusion(self, state: RAGState):
        """Execute document retrieval while applying strict ID-level exclusions from history."""
        logger.info("Executing specialized retrieval with database-level exclusion")

        query = state.get("rewritten_query") or state.get("question")
        metadata = state.get("retriever_metadata", {})

        exclude_set = set(str(doc_id) for doc_id in metadata.get("exclude_ids", []))

        history = state.get("documents_history", [])

        exclude_set = set()

        for doc_entry in history:
            doc_id = doc_entry.get("document_id")
            if doc_id is not None:
                exclude_set.add(str(doc_id))

        user_info = state.get("user", {})

        exclude_ids = list(exclude_set) if exclude_set else None

        search_filters = {
            "is_professional": user_info.get("is_professional"),
            "is_common": user_info.get("is_common"),
            "exclude_ids": exclude_ids,
        }

        logger.info(f"search_filters={search_filters}")

        docs = await self.retriever.aretrieve(
            query=query,
            filters=search_filters,
        )

        logger.info(f"Database returned {len(docs)} new documents")

        return {
            "documents": docs,
            "retriever_metadata": {
                **metadata,
                "exclude_ids": exclude_ids,
            },
        }

    @observability.time_it_async("Update User Profile After Clarification")
    async def update_profile_after_clarification(self, state: RAGState):
        """Extract and persist new user information provided during a clarification dialogue."""
        user_id = state.get("user", {}).get("user_id")
        if not user_id:
            return {}

        messages = state.get("messages", [])

        last_user = None
        last_assistant = None

        for m in reversed(messages):
            if isinstance(m, HumanMessage) and not last_user:
                last_user = m.content
            if isinstance(m, AIMessage) and not last_assistant:
                last_assistant = m.content
            if last_user and last_assistant:
                break

        if not last_user:
            return {}

        current_profile = state.get("user", {}).get("profile") or {}

        updated_profile = await self.chains.profile_updater.ainvoke(
            {
                "current_profile": json.dumps(current_profile),
                "latest_message": last_user,
            }
        )

        await self.profile_repo.save(user_id, updated_profile)

        return {"user_profile": updated_profile}
