import streamlit as st

# Configure the global appearance and layout of the Streamlit application
st.set_page_config(
    page_title="RAG Agent",
    layout="wide",
)

# Define the navigation menu and page routing
pg = st.navigation(
    [
        st.Page("pages/chat.py", title="Chat"),
        st.Page("pages/documents.py", title="Documents"),
    ]
)

# Execute the routing logic to render the selected page
pg.run()
