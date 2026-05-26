import asyncio

import streamlit as st

from components.chat_messages import render_messages
from components.sidebar import render_sidebar
from core.deps import get_context

# Initialize application context and dependencies
ctx = get_context()

st.title("RAG Chat")

# Render sidebar and handle state reset
sidebar = render_sidebar()

if sidebar["clear"]:
    # Reset the conversation history.
    st.session_state.messages = []
    st.rerun()

# Ensure message history exists in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing conversation
render_messages(st.session_state.messages)

# Main chat interaction loop.
if prompt := st.chat_input("Ask something..."):

    # Display user message in the chat container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare data for the RAG agent
    payload = {
        "question": prompt,
        "messages": st.session_state.messages,
        "user": {"user_id": sidebar["user_id"]},
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()

        with st.spinner("Thinking..."):
            try:
                # Execute the API request synchronously
                response = asyncio.run(ctx.api_service.ask_question(payload=payload))

                if response.status_code == 200:
                    data = response.json()

                    st.session_state.messages = data["messages"]

                    # Extract and display the generated answer
                    answer = data.get("answer", "")
                    placeholder.markdown(answer)

                else:
                    st.error(f"API Error: {response.status_code}")

            except Exception as e:
                st.error(f"Connection error: {e}")
