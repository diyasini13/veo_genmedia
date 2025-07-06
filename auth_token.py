import streamlit as st
import os

def get_authenticated_user():
    # Check if user info is already in session state
    if st.session_state.get('user_info_loaded', False):
        return st.session_state.user_info

    # Use Streamlit's built-in OAuth for Google
    # This will trigger the Google login flow if the user is not authenticated.
    # Streamlit automatically uses the configured OAuth details (from environment variables on Cloud Run)
    if not st.session_state.get('streamlit_oauth_logged_in'):
        st.title("Welcome to GenMedia Studio!")
        st.write("Please sign in with your Google account to continue.")
        
        # This button triggers the Google login flow using Streamlit's internal OAuth
        if st.button("Login with Google"):
            st.login("google") # Direct Streamlit to use the 'google' OAuth provider
        st.stop() # Stop execution until the user is authenticated

    # If the app execution reaches here, it means Streamlit has authenticated the user.
    # The user object is available through st.user
    user = st.user
    if user and user.is_logged_in:
        user_data = {
            "name": user.name,
            "email": user.email,
            
            # Add more user attributes if Google provides them and you need them
        }
        st.session_state.user_info = user_data
        st.session_state.user_info_loaded = True # Mark as loaded to prevent re-authentication checks on reruns
        st.session_state.streamlit_oauth_logged_in = True # Keep track that Streamlit has logged in
        return user_data
    else:
        # Should not typically happen if st.stop() works as expected before this point
        return None

def display_login_button():
    # This function might not be strictly necessary if get_authenticated_user handles the login UI.
    pass