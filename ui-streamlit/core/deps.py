import streamlit as st

from bootstrap import bootstrap


def get_context():
    """
    Retrieve or initialize the application context within the Streamlit session.

    This function implements a singleton-like pattern using `st.session_state` 
    to ensure that the application context is bootstrapped only once per session.

    Returns
    -------
    Any
        The application context object returned by the `bootstrap` function, 
        stored in the session state.
    """
    # Check if the context is already present in the current session
    if "ctx" not in st.session_state:
        st.session_state.ctx = bootstrap()

    return st.session_state.ctx
