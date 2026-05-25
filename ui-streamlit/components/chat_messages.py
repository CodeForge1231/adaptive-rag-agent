import streamlit as st


def render_messages(messages):
    """
    Render the chat history in the Streamlit interface.

    Parameters
    ----------
    messages : list of dict
        A list of message dictionaries. Each dictionary must contain:
        - 'role' (str): The entity that sent the message (e.g., 'user', 'assistant').
        - 'content' (str): The markdown-formatted text to be displayed.

    Returns
    -------
    None
    """
    for message in messages:
        # Create a chat container context based on the message role
        with st.chat_message(message["role"]):
            # Render the message text using markdown formatting
            st.markdown(message["content"])
