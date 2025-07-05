import streamlit as st
from google.oauth2 import id_token
from google.auth.transport import requests
import os
from dotenv import load_dotenv

load_dotenv()

# It's better to load the client ID from an environment variable
# for security and flexibility. You can set this in your .env file
# or as a Streamlit secret.
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "536936801426-bq8u0r27na73gv23nu88jrda15537qk4.apps.googleusercontent.com")

def verify_token(token: str) -> dict | None:
    """Verifies a Google OAuth2 ID token.

    Args:
        token: The ID token to verify.

    Returns:
        A dictionary containing the decoded token information if valid,
        otherwise None.
    """
    if not GOOGLE_CLIENT_ID:
        st.error("GOOGLE_CLIENT_ID is not set. Authentication cannot be verified.")
        return None
        
    try:
        # The 'aud' (audience) check is automatically performed by verify_oauth2_token
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), GOOGLE_CLIENT_ID)
        return idinfo
    except ValueError as e:
        st.error(f"Token verification failed: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during token verification: {e}")
        return None

def get_authenticated_user() -> dict | None:
    """
    Checks for a token in the query parameters, verifies it, and returns
    the user information if the token is valid.
    
    Manages user info in session state.
    """
    if 'user_info' in st.session_state:
        return st.session_state.user_info

    # st.query_params is experimental and returns a list.
    token_list = st.query_params.get("token", [])
    token = token_list[0] if token_list else None
    
    if token:
        user_info = verify_token(token)
        if user_info:
            st.session_state.user_info = user_info
            # Clear the token from the URL for a cleaner user experience and to prevent reuse.
            st.query_params.clear()
            return user_info
    return None

def display_login_button():
    """Displays the Google Sign-In button using HTML and JavaScript."""
    if not GOOGLE_CLIENT_ID:
        st.error("Google Client ID is not configured. Cannot display login button.")
        return

    # The redirect logic is now handled entirely by JavaScript using the current
    # window location. This makes it work seamlessly for both local development
    # and deployment without changing the code.
    #
    # You just need to ensure that BOTH `http://localhost:8501` AND your deployed URL
    # (e.g., https://veo-genmedia-536936801426.us-central1.run.app) are added to the
    # "Authorized JavaScript origins" and "Authorized redirect URIs" in your
    # Google Cloud Console's OAuth Client ID settings (see Step 2).

    st.html(f'''
        <script src="https://accounts.google.com/gsi/client" async defer></script>
        <script>
        function handleCredentialResponse(response) {{
            const id_token = response.credential;
            // Use the current window's origin to build the redirect URL.
            // This works for both localhost and deployed environments.
            const redirect_url = new URL(window.location.origin);
            redirect_url.searchParams.set('token', id_token);
            window.location.href = redirect_url.href;
        }}
        window.onload = function () {{
            google.accounts.id.initialize({{
                client_id: "{GOOGLE_CLIENT_ID}",
                callback: handleCredentialResponse
            }});
            // Render the button in the center.
            const parent = document.getElementById("buttonDiv");
            google.accounts.id.renderButton(parent, {{ theme: "outline", size: "large", text: "signin_with" }});
            parent.style.display = 'flex';
            parent.style.justifyContent = 'center';
        }};
        </script>
        <div id="buttonDiv"></div>
    ''')
