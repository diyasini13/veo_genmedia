# auth_token.py - Revised for minimal scopes
import streamlit as st
from google_auth_oauthlib import flow
import os
import google.auth.transport.requests

# Define the absolute minimal scopes for a basic OpenID Connect sign-in.
# 'openid' is typically required for Google to issue an ID token that contains
# basic user information like email and name.
SCOPES = ['openid'] 

def get_authenticated_user():
    """
    Checks if a user is authenticated and returns their information.
    Handles the OAuth callback if a 'code' is present in the URL parameters.
    """
    # Initialize user_info and credentials in session_state if not present
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'credentials' not in st.session_state:
        st.session_state.credentials = None
    if 'oauth_state' not in st.session_state:
        st.session_state.oauth_state = None

    # If user info is already in session, return it
    if st.session_state.user_info:
        return st.session_state.user_info

    # Check for OAuth callback parameters in the URL
    query_params = st.query_params
    code = query_params.get("code")
    state = query_params.get("state")

    if code and state:
        # Check if the state matches to prevent CSRF
        if state == st.session_state.oauth_state:
            try:
                client_id = st.secrets["google_oauth"]["client_id"]
                client_secret = st.secrets["google_oauth"]["client_secret"]

                # Determine redirect URI for local or deployed environment
                
                redirect_uri = "https://veo-genmedia-536936801426.us-central1.run.app"

                flow_instance = flow.Flow.from_client_config(
                    client_config={
                        "web": {
                            "client_id": client_id,
                            "client_secret": client_secret,
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                            "redirect_uris": [redirect_uri],
                            "javascript_origins": [redirect_uri]
                        }
                    },
                    scopes=SCOPES # Use the minimal SCOPES defined
                )
                flow_instance.redirect_uri = redirect_uri

                # Exchange the authorization code for tokens
                # This will also get the ID token if 'openid' scope is present
                flow_instance.fetch_token(code=code)
                st.session_state.credentials = flow_instance.credentials

                if st.session_state.credentials and st.session_state.credentials.valid:
                    user_info = {}
                    # The ID token should contain basic profile info if 'openid' is requested
                    if hasattr(flow_instance.credentials, 'id_token') and flow_instance.credentials.id_token:
                        id_token_data = flow_instance.credentials.id_token
                        user_info = {
                            "email": id_token_data.get('email'),
                            "name": id_token_data.get('name'),
                            "picture": id_token_data.get('picture', 'https://www.gstatic.com/images/branding/product/2x/avatar_square_grey_24dp.png')
                        }
                    else:
                        # Fallback if id_token is not available despite 'openid' (unlikely with Google)
                        # This might require an additional API call, but we'll try to rely on id_token
                        st.warning("ID token not found. User information may be limited.")

                    st.session_state.user_info = user_info
                    
                    st.experimental_set_query_params() 
                    st.rerun() 
                    return st.session_state.user_info
                else:
                    st.error("Authentication failed: Invalid or expired credentials.")
                    st.session_state.credentials = None
                    st.session_state.user_info = None
                    st.session_state.oauth_state = None
                    st.experimental_set_query_params() 
                    return None

            except Exception as e:
                st.error(f"Error during authentication callback: {e}")
                st.session_state.credentials = None
                st.session_state.user_info = None
                st.session_state.oauth_state = None
                st.experimental_set_query_params() 
                return None
        else:
            st.error("Authentication failed: State mismatch. Possible CSRF attack.")
            st.session_state.credentials = None
            st.session_state.user_info = None
            st.session_state.oauth_state = None
            st.experimental_set_query_params() 
            return None
    
    return None

def display_login_button():
    """
    Displays the Google Sign-In button and initiates the OAuth flow.
    """
    try:
        client_id = st.secrets["google_oauth"]["client_id"]
        client_secret = st.secrets["google_oauth"]["client_secret"]

        
        redirect_uri = "https://veo-genmedia-536936801426.us-central1.run.app"

        flow_instance = flow.Flow.from_client_config(
            client_config={
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "redirect_uris": [redirect_uri],
                    "javascript_origins": [redirect_uri]
                }
            },
            scopes=SCOPES # Use the minimal SCOPES defined
        )
        flow_instance.redirect_uri = redirect_uri

        # We request 'offline' access if you want a refresh token,
        # which allows your app to get new access tokens without the user re-authenticating.
        # If you truly want NO persistent access beyond the immediate session,
        # you could remove access_type='offline'.
        authorization_url, state = flow_instance.authorization_url(
            access_type='offline', # Request refresh token (optional based on your need for persistence)
            include_granted_scopes='true'
        )
        st.session_state['oauth_state'] = state 

        st.link_button("Sign in with Google", url=authorization_url)

    except Exception as e:
        st.error(f"Error setting up Google Sign-In: {e}")
        st.info("Please ensure your `secrets.toml` file contains `client_id` and `client_secret` under `[google_oauth]`, and your Google Cloud project's OAuth 2.0 client ID is correctly configured with `Authorized JavaScript origins` and `Authorized redirect URIs`.")