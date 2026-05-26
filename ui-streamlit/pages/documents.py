import asyncio
import uuid

import streamlit as st

from components.document_form import (
    render_document_form,
)
from components.sidebar import render_sidebar
from core.deps import get_context

# Initialize application context and dependencies
ctx = get_context()

st.title("Add Documents")

# Render UI components
sidebar = render_sidebar()
form = render_document_form()

# Handle the 'Add Document' button click event.
if st.button("Add Document"):

    # Prepare document data structure
    payload = {
        "id": str(uuid.uuid4()),
        "text": form["text"],
        "metadata": form["metadata"],
    }

    try:
        # Execute asynchronous API request
        response = asyncio.run(
            ctx.api_service.add_document(
                payload=payload,
            )
        )
        # Evaluate API response status
        if response.status_code == 200:
            st.success("Document added successfully")
        # Handle non-200 status codes from the API
        else:
            st.error(f"API Error: {response.status_code}")

    except Exception as e:
        st.error(f"Connection error: {e}")
