import streamlit as st
from google.cloud import aiplatform, storage # Import storage
import vertexai
from vertexai.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part
from google.genai.types import GenerateVideosConfig, Image  # Keep this import
from google import genai
import time
import os
from PIL import Image as PILImage # Alias Image from PIL to avoid conflict with google.genai.types.Image
import requests # Still useful for general HTTP but we'll use storage for GCS
import uuid # For generating unique filenames
from google.genai import types
import io # Import io for handling image bytes from upload

# --- Configuration ---

# NOTE: Hardcoding credentials is not recommended for production environments.
# It's better to use a more secure method like service account keys
# managed by your cloud environment or Streamlit's secrets management.
# Ensure this path is correct for your environment.
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/diyasini/Desktop/Live Demo page/GenMedia/key-svc-gen-ai.json.json'
os.environ['GOOGLE_CLOUD_PROJECT'] = 'genail300'
os.environ['GCP_REGION'] = 'us-central1'
project_id = "genail300"
gcs_bucket_name = "veo_testing_hp" # Replace with your GCS bucket name.
                                                 # This bucket will be used for both input images and output videos.
                                                 # Ensure it exists and your service account has read/write permissions.

# Initialize Vertex AI. The SDK will authenticate automatically.
vertexai.init(project=project_id, location="us-central1")

# Initialize the genai client, explicitly pointing it to Vertex AI
client = genai.Client(vertexai=True, project=project_id, location=os.environ['GCP_REGION'])

# Initialize Google Cloud Storage client
storage_client = storage.Client(project=project_id)

# --- Model Helper Functions ---

def refine_prompt_with_gemini(user_prompt: str, for_video: bool = False) -> str:
    """Uses Gemini on Vertex AI to refine a user's prompt for image or video generation."""
    model = GenerativeModel("gemini-2.0-flash") # Using a faster model for refinement
    
    if for_video:
        refinement_prompt = f"""
        You are an expert prompt engineer for text-to-video models.
        Your task is to refine the following user prompt to make it more descriptive, vivid, and detailed for generating a high-quality, engaging video from a static image.
        Focus on adding specific details about camera movements (e.g., pan, zoom, tilt), character actions, environmental changes, and dynamic elements.
        Return only the refined prompt, without any additional text or explanation.

        User Prompt: "{user_prompt}"
        """
    else: # For image generation
        refinement_prompt = f"""
        You are an expert prompt engineer for text-to-image models.
        Your task is to refine the following user prompt to make it more descriptive, vivid, and detailed for generating a high-quality, photorealistic image.
        Add specific details about the subject, environment, lighting, camera angle, and overall style.
        Return only the refined prompt, without any additional text or explanation.

        User Prompt: "{user_prompt}"
        """
    
    try:
        response = model.generate_content(refinement_prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"An error occurred with Gemini: {e}")
        return ""

def generate_image_with_imagen(prompt: str) -> bytes:
    """Generates an image using Imagen on Vertex AI and returns its bytes."""
    try:
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="16:9", # or "1:1", "9:16"
            safety_filter_level="block_some",
            person_generation="allow_adult"
        )
        
        # Get image bytes from the first generated image
        image_bytes = response.images[0]._image_bytes
        return image_bytes
    except Exception as e:
        st.error(f"An error occurred with Imagen: {e}")
        return None

def upload_bytes_to_gcs(bucket_name: str, blob_name: str, data: bytes, content_type: str = "image/png") -> str:
    """Uploads bytes data to GCS and returns the gs:// URI."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{bucket_name}/{blob_name}"
    except Exception as e:
        st.error(f"Failed to upload image to GCS: {e}")
        return None

def generate_video_with_veo(image_bytes: bytes, video_prompt: str) -> bytes | None:
    """
    Generates a video from an image and a prompt using the Veo model on Vertex AI.
    This function handles the long-running operation and downloads the video from GCS
    for display in Streamlit.
    """
    st.info("🎬 Calling the Veo model... This is a long-running operation and may take several minutes.")

    # 1. Upload the input image (either Imagen generated or user uploaded) to GCS first
    unique_image_filename = f"veo_input_image_{uuid.uuid4()}.png"
    input_image_gcs_uri = upload_bytes_to_gcs(gcs_bucket_name, unique_image_filename, image_bytes)
    
    if not input_image_gcs_uri:
        return None # Exit if GCS upload failed

    st.info(f"Uploaded input image to GCS: {input_image_gcs_uri}")

    # Generate a unique GCS output URI for each video request
    timestamp = int(time.time())
    output_video_gcs_uri = f"gs://{gcs_bucket_name}/veo_output_{timestamp}.mp4"

    try:
        # Create a genai.types.Image object using the GCS URI of the uploaded image
        genai_image = types.Image(
            gcs_uri=input_image_gcs_uri,
            mime_type="image/png",
        )

        operation = client.models.generate_videos(
            model="veo-2.0-generate-001",
            prompt=video_prompt, # Use the specific video prompt
            image=genai_image, # Use the correctly formed genai_image object
            config=GenerateVideosConfig(
                aspect_ratio="16:9",
                number_of_videos=1,
                duration_seconds=8,
                output_gcs_uri=output_video_gcs_uri,
            ),
        ) 

        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        while not operation.done:
            time.sleep(15) # Poll every 15 seconds
            # IMPORTANT: Use operation.name to get the updated status for a long-running operation
            operation = client.operations.get(operation) 
            
            elapsed_time = time.time() - start_time
            status_text.text(f"Video generation in progress... Elapsed time: {int(elapsed_time)}s")
            
            # Simple progress simulation based on time (Veo doesn't provide real-time progress)
            # Adjust 120 (seconds) based on typical Veo generation time for your previews
            progress = min(int(elapsed_time / 120 * 100), 99) 
            progress_bar.progress(progress)
            
            # print(f"Operation status: {operation.done}, {operation.error}, {operation.response}") # For debugging

        progress_bar.progress(100)
        status_text.text("Video generation complete!")

        if operation.response and operation.result and operation.result.generated_videos:
            video_uri = operation.result.generated_videos[0].video.uri
            st.success(f"Video generated to GCS: {video_uri}")

            # 2. Download the video from GCS using google.cloud.storage client
            st.info("Downloading video for display...")
            try:
                video_blob_name = video_uri.replace(f"gs://{gcs_bucket_name}/", "")
                bucket = storage_client.bucket(gcs_bucket_name)
                blob = bucket.blob(video_blob_name)
                video_bytes = blob.download_as_bytes()
                
                st.success("Video downloaded successfully!")
                return video_bytes
            except Exception as e:
                st.error(f"Failed to download video from GCS: {e}")
                st.warning("Please ensure the GCS bucket has appropriate permissions for downloading.")
                return None
        else:
            st.error("Veo operation completed but no video URI found in the response.")
            if operation.error:
                st.error(f"Veo API error: {operation.error.message}")
            return None

    except Exception as e:
        st.error(f"An error occurred with Veo: {e}")
        return None

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Gemini & Imagen & Veo Studio")

st.title("✨ AI-Powered Creative Studio")
st.markdown("Create stunning visuals and videos with Gemini, Imagen, and Veo.")

# Initialize session state variables
if 'refined_prompt' not in st.session_state:
    st.session_state.refined_prompt = ""
if 'image_bytes' not in st.session_state: # Stores Imagen generated image bytes
    st.session_state.image_bytes = None
if 'uploaded_image_bytes' not in st.session_state: # Stores user uploaded image bytes
    st.session_state.uploaded_image_bytes = None
if 'video_bytes' not in st.session_state:
    st.session_state.video_bytes = None
# Initialize user_entered_prompt in session state for text_area control for IMAGE
if 'user_entered_prompt' not in st.session_state:
    st.session_state.user_entered_prompt = ""
# New: Initialize user_entered_prompt and refined_prompt for VIDEO
if 'veo_user_prompt' not in st.session_state:
    st.session_state.veo_user_prompt = ""
if 'veo_refined_prompt' not in st.session_state:
    st.session_state.veo_refined_prompt = ""


# --- Step 1: Craft Your Image Prompt ---
st.header("Step 1: Craft and Refine Your **Image** Prompt")
st.markdown("Start by describing your visual idea. Gemini can help you make it more detailed for better image results.")

col1_img_prompt, col2_img_prompt = st.columns(2)

with col1_img_prompt:
    user_prompt_input_current_value = st.text_area(
        "Enter your initial idea for the image:", 
        height=150, 
        key="user_prompt_input",
        value=st.session_state.user_entered_prompt, # Links to session state
        help="Describe the scene, subject, and desired style for the base image."
    )
    
    # Update session state if the user manually changes the text area
    st.session_state.user_entered_prompt = user_prompt_input_current_value 

    if st.button("🚀 Refine Image Prompt with Gemini", use_container_width=True, type="primary"):
        if st.session_state.user_entered_prompt: 
            with st.spinner("Gemini is refining your image prompt..."):
                refined_text = refine_prompt_with_gemini(st.session_state.user_entered_prompt, for_video=False)
                
                if refined_text:
                    st.session_state.refined_prompt = refined_text
                    # Update user_entered_prompt to display the refined text in the text_area
                    st.session_state.user_entered_prompt = refined_text 
            
            # Clear generated/uploaded content and video prompts when image prompt is refined
            st.session_state.image_bytes = None 
            st.session_state.uploaded_image_bytes = None 
            st.session_state.video_bytes = None 
            st.session_state.veo_user_prompt = "" # Clear video prompt
            st.session_state.veo_refined_prompt = "" # Clear video refined prompt
        else:
            st.warning("Please enter an initial idea for the image to refine.")

with col2_img_prompt:
    if st.session_state.refined_prompt:
        st.markdown("**Gemini's Refined Image Prompt:**")
        st.info(st.session_state.refined_prompt)
    else:
        st.info("Your refined image prompt, enhanced by Gemini for optimal generation, will appear here.")

# --- Step 2: Image Generation or Upload ---
st.header("Step 2: Generate or Upload a Base Image")
st.markdown("This image will serve as the static foundation for your video.")

tab_generate, tab_upload = st.tabs(["Generate with Imagen", "Upload Your Image"])

with tab_generate:
    # Use refined prompt if available, otherwise the raw user prompt for image generation
    prompt_to_use_for_imagen = st.session_state.refined_prompt if st.session_state.refined_prompt else st.session_state.user_entered_prompt

    if st.button("🎨 Generate Image with Imagen", use_container_width=True, disabled=not prompt_to_use_for_imagen, key="generate_image_button"):
        if prompt_to_use_for_imagen:
            with st.spinner("Imagen is generating your image... This may take a moment."):
                st.session_state.image_bytes = generate_image_with_imagen(prompt_to_use_for_imagen)
                st.session_state.uploaded_image_bytes = None # Clear uploaded image if new image is generated
                st.session_state.video_bytes = None # Reset video if a new image is generated
                st.session_state.veo_user_prompt = "" # Clear video prompt
                st.session_state.veo_refined_prompt = "" # Clear video refined prompt
        else:
            st.warning("Please provide a prompt (in Step 1) to generate an image.")
    
    if st.session_state.image_bytes:
        # Display Imagen generated image, you can also set a fixed width here if desired
        st.image(st.session_state.image_bytes, caption="Generated by Imagen", use_column_width=True)
    else:
        st.info("Your generated image will appear here.")

with tab_upload:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

    if uploaded_file is not None:
        # Read the image bytes
        uploaded_image_bytes_raw = uploaded_file.getvalue()
        
        # Display the uploaded image with a defined width
        st.image(uploaded_image_bytes_raw, caption="Uploaded Image", width=200) # Reduced width for display
        
        # Update session state
        st.session_state.uploaded_image_bytes = uploaded_image_bytes_raw
        st.session_state.image_bytes = None # Clear generated image if new image is uploaded
        st.session_state.video_bytes = None # Reset video if a new image is uploaded
        st.session_state.veo_user_prompt = "" # Clear video prompt
        st.session_state.veo_refined_prompt = "" # Clear video refined prompt

    if st.session_state.uploaded_image_bytes:
        st.success("Image uploaded successfully! Proceed to Step 3.")
    else:
        st.info("Upload an image (JPG, JPEG, PNG) to use for video generation.")


# --- Step 3: Animate Your Image with Veo ---
st.header("Step 3: Animate Your Image with Veo")
st.markdown("Provide a **separate prompt** to guide the video generation. Describe the motion, actions, and camera movements you envision.")

# Determine which image to use for video generation
image_for_veo = None
if st.session_state.image_bytes:
    image_for_veo = st.session_state.image_bytes
    st.info("Using the **Imagen-generated image** for video creation.")
elif st.session_state.uploaded_image_bytes:
    image_for_veo = st.session_state.uploaded_image_bytes
    st.info("Using the **uploaded image** for video creation.")
else:
    st.warning("Please generate or upload an image in Step 2 to proceed with video generation.")


col1_vid_prompt, col2_vid_prompt = st.columns(2)

with col1_vid_prompt:
    veo_user_prompt_current_value = st.text_area(
        "Enter your idea for the video's motion and story:", 
        height=150, 
        key="veo_user_prompt_input",
        value=st.session_state.veo_user_prompt, # Links to session state
        help="Describe what should happen in the video, including character actions or camera movements (e.g., 'camera pans left slowly', 'a car drives by')."
    )
    st.session_state.veo_user_prompt = veo_user_prompt_current_value

    if st.button("🚀 Refine Video Prompt with Gemini", use_container_width=True, type="primary", key="refine_video_prompt_button"):
        if st.session_state.veo_user_prompt:
            with st.spinner("Gemini is refining your video prompt..."):
                refined_veo_text = refine_prompt_with_gemini(st.session_state.veo_user_prompt, for_video=True)
                if refined_veo_text:
                    st.session_state.veo_refined_prompt = refined_veo_text
                    st.session_state.veo_user_prompt = refined_veo_text # Update text area with refined prompt
            st.session_state.video_bytes = None # Clear previous video if prompt is refined
        else:
            st.warning("Please enter an initial idea for the video to refine.")

with col2_vid_prompt:
    if st.session_state.veo_refined_prompt:
        st.markdown("**Gemini's Refined Video Prompt:**")
        st.info(st.session_state.veo_refined_prompt)
    elif st.session_state.veo_user_prompt:
        st.markdown("**Your Video Prompt:**")
        st.info(st.session_state.veo_user_prompt)
    else:
        st.info("Your refined video prompt will appear here.")

# Determine the final prompt to send to Veo: refined video prompt > user video prompt > refined image prompt > user image prompt
final_veo_prompt = (
    st.session_state.veo_refined_prompt
    or st.session_state.veo_user_prompt
    # or st.session_state.refined_prompt
    # or st.session_state.user_entered_prompt
)

st.markdown("---") # Separator before the generation button

if st.button("🎬 Generate Video from Image", use_container_width=True, 
             disabled=not (image_for_veo and final_veo_prompt), key="generate_video_button"):
    if not image_for_veo:
        st.warning("Please generate or upload an image in Step 2 first.")
    elif not final_veo_prompt:
        st.warning("Please provide a prompt for video generation in Step 3.")
    else:
        with st.spinner("Veo is animating your image... This can take several minutes."):
            st.session_state.video_bytes = generate_video_with_veo(image_for_veo, final_veo_prompt)

if st.session_state.video_bytes:
    st.subheader("Your Generated Video")
    st.video(st.session_state.video_bytes)
else:
    st.info("Your generated video will appear here after Veo finishes processing.")