import streamlit as st

from constants.metadata import (
    AGE_GROUPS,
    LEVELS,
    TOPICS,
)


def render_document_form():
    """
    Render a form in the Streamlit UI for document data entry.

    This function creates an interface for inputting document content and 
    selecting associated metadata through text areas and selection boxes.

    Returns
    -------
    dict
        A structured dictionary containing the user input:
        - 'text' (str): The raw text content of the document.
        - 'metadata' (dict): A nested dictionary with classification details:
            - 'age_group' (str): Selected target audience category.
            - 'level' (str): Selected proficiency level.
            - 'topic' (str): Selected subject matter.
    """

    # Main input area for the document body
    text = st.text_area("Document text", height=250)

    # Metadata selection fields using predefined constants
    age_group = st.selectbox("Age group", AGE_GROUPS)
    level = st.selectbox("Level", LEVELS)
    topic = st.selectbox("Topic", TOPICS)

    # Return structured data for further processing or API transmission
    return {
        "text": text,
        "metadata": {
            "age_group": age_group,
            "level": level,
            "topic": topic,
        },
    }
