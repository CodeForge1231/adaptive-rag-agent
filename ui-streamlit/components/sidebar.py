import streamlit as st


def render_sidebar():
    """
    Render the sidebar UI components for application settings.

    Returns
    -------
    dict
        A dictionary containing the state of the sidebar widgets:
        - 'user_id' (str): The unique identifier entered by the user.
        - 'clear' (bool): True if the 'Clear Chat' button was clicked, 
          False otherwise.
    """
    with st.sidebar:
        st.header("Settings")

        # Input field for defining the target user context
        user_id = st.text_input("User ID", "1")

        # Action button to trigger chat history reset
        clear = st.button("Clear Chat")

    return {
        "user_id": user_id,
        "clear": clear,
    }
