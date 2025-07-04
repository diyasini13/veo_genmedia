import streamlit as st
from google.cloud import aiplatform, storage
import vertexai
from vertexai.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part
from google.genai.types import GenerateVideosConfig, Image
from google import genai
import time
import os
from PIL import Image as PILImage
import requests
import uuid
from google.genai import types
import io

# --- Configuration ---
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/diyasini/Desktop/Live Demo page/GenMedia/key-svc-gen-ai.json.json'
os.environ['GOOGLE_CLOUD_PROJECT'] = 'genail300'
os.environ['GCP_REGION'] = 'us-central1'
project_id = "genail300"
gcs_bucket_name = "veo_testing_hp"

vertexai.init(project=project_id, location="us-central1")
client = genai.Client(vertexai=True, project=project_id, location=os.environ['GCP_REGION'])
storage_client = storage.Client(project=project_id)

# --- Model Helper Functions ---
def refine_prompt_with_gemini(user_prompt: str, for_video: bool = False) -> str:
    """Uses Gemini on Vertex AI to refine a user's prompt for image or video generation."""
    model = GenerativeModel("gemini-2.0-flash")
    
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
            aspect_ratio="16:9",
            safety_filter_level="block_some",
            person_generation="allow_adult"
        )
        
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

def generate_video_with_veo(input_type: str, input_content: bytes | str, video_prompt: str) -> bytes | None:
    """
    Generates a video using the Veo model on Vertex AI.
    This function handles the long-running operation and downloads the video from GCS
    for display in Streamlit. It can accept either image bytes or a text prompt as input.
    """
    # st.info("🎬 Calling the Veo model... This is a long-running operation and may take several minutes.")

    genai_image = None
    if input_type == "image":
        unique_image_filename = f"veo_input_image_{uuid.uuid4()}.png"
        input_image_gcs_uri = upload_bytes_to_gcs(gcs_bucket_name, unique_image_filename, input_content)
        
        if not input_image_gcs_uri:
            return None

        st.info(f"Uploaded input image to GCS: {input_image_gcs_uri}")
        genai_image = types.Image(
            gcs_uri=input_image_gcs_uri,
            mime_type="image/png",
        )

    timestamp = int(time.time())
    output_video_gcs_uri = f"gs://{gcs_bucket_name}/veo_output_{timestamp}.mp4"

    try:
        if input_type == "image":
            operation = client.models.generate_videos(
                model="veo-2.0-generate-001",
                prompt=video_prompt,
                image=genai_image,
                config=GenerateVideosConfig(
                    aspect_ratio="16:9",
                    number_of_videos=1,
                    duration_seconds=8,
                    output_gcs_uri=output_video_gcs_uri,
                ),
            ) 
        else: # Text to Video
            operation = client.models.generate_videos(
                model="veo-2.0-generate-001",
                prompt=video_prompt,
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
            time.sleep(15)
            operation = client.operations.get(operation) 
            
            elapsed_time = time.time() - start_time
            status_text.text(f"Video generation in progress... Elapsed time: {int(elapsed_time)}s")
            
            progress = min(int(elapsed_time / 120 * 100), 99) 
            progress_bar.progress(progress)
            
        progress_bar.progress(100)
        status_text.text("Video generation complete!")

        if operation.response and operation.result and operation.result.generated_videos:
            video_uri = operation.result.generated_videos[0].video.uri
            # st.success(f"Video generated to GCS: {video_uri}")

            # st.info("Downloading video for display...")
            try:
                video_blob_name = video_uri.replace(f"gs://{gcs_bucket_name}/", "")
                bucket = storage_client.bucket(gcs_bucket_name)
                blob = bucket.blob(video_blob_name)
                video_bytes = blob.download_as_bytes()
                
                # st.success("Video downloaded successfully!")
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
if 'image_bytes' not in st.session_state:
    st.session_state.image_bytes = None
if 'uploaded_image_bytes' not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if 'video_bytes' not in st.session_state:
    st.session_state.video_bytes = None
if 'user_entered_prompt' not in st.session_state:
    st.session_state.user_entered_prompt = ""
if 'veo_user_prompt' not in st.session_state:
    st.session_state.veo_user_prompt = ""
if 'veo_refined_prompt' not in st.session_state:
    st.session_state.veo_refined_prompt = ""
if 'text_to_video_user_prompt' not in st.session_state:
    st.session_state.text_to_video_user_prompt = ""
if 'text_to_video_refined_prompt' not in st.session_state:
    st.session_state.text_to_video_refined_prompt = ""

# --- Navigation Bar (Using st.tabs) ---
tab_image_to_video, tab_text_to_video = st.tabs(["🖼️ Image to Video Generation", "📝 Text to Video Generation"])

with tab_image_to_video:
    # --- Step 1: Craft Your Image Prompt ---
    st.header("Step 1: Craft and Refine Your **Image** Prompt")
    st.markdown("Start by describing your visual idea. Gemini can help you make it more detailed for better image results.")

    col1_img_prompt, col2_img_prompt = st.columns(2)

    with col1_img_prompt:
        user_prompt_input_current_value = st.text_area(
            "Enter your initial idea for the image:", 
            height=150, 
            key="user_prompt_input",
            value=st.session_state.user_entered_prompt,
            help="Describe the scene, subject, and desired style for the base image."
        )
        
        st.session_state.user_entered_prompt = user_prompt_input_current_value 

        if st.button("🚀 Refine Image Prompt with Gemini", use_container_width=True, type="primary"):
            if st.session_state.user_entered_prompt: 
                with st.spinner("Gemini is refining your image prompt..."):
                    refined_text = refine_prompt_with_gemini(st.session_state.user_entered_prompt, for_video=False)
                    
                    if refined_text:
                        st.session_state.refined_prompt = refined_text
                        st.session_state.user_entered_prompt = refined_text 
                
                st.session_state.image_bytes = None 
                st.session_state.uploaded_image_bytes = None 
                st.session_state.video_bytes = None 
                st.session_state.veo_user_prompt = "" 
                st.session_state.veo_refined_prompt = "" 
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

    tab_generate_inner, tab_upload_inner = st.tabs(["Generate with Imagen", "Upload Your Image"])

    with tab_generate_inner:
        prompt_to_use_for_imagen = st.session_state.refined_prompt if st.session_state.refined_prompt else st.session_state.user_entered_prompt

        if st.button("🎨 Generate Image with Imagen", use_container_width=True, disabled=not prompt_to_use_for_imagen, key="generate_image_button"):
            if prompt_to_use_for_imagen:
                with st.spinner("Imagen is generating your image... This may take a moment."):
                    st.session_state.image_bytes = generate_image_with_imagen(prompt_to_use_for_imagen)
                    st.session_state.uploaded_image_bytes = None
                    st.session_state.video_bytes = None
                    st.session_state.veo_user_prompt = ""
                    st.session_state.veo_refined_prompt = ""
            else:
                st.warning("Please provide a prompt (in Step 1) to generate an image.")
        
        if st.session_state.image_bytes:
            st.image(st.session_state.image_bytes, caption="Generated by Imagen", use_column_width=True)
        else:
            st.info("Your generated image will appear here.")

    with tab_upload_inner:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

        if uploaded_file is not None:
            uploaded_image_bytes_raw = uploaded_file.getvalue()
            st.image(uploaded_image_bytes_raw, caption="Uploaded Image", width=200)
            st.session_state.uploaded_image_bytes = uploaded_image_bytes_raw
            st.session_state.image_bytes = None
            st.session_state.video_bytes = None
            st.session_state.veo_user_prompt = ""
            st.session_state.veo_refined_prompt = ""

        if st.session_state.uploaded_image_bytes:
            st.success("Image uploaded successfully! Proceed to Step 3.")
        else:
            st.info("Upload an image (JPG, JPEG, PNG) to use for video generation.")

    # --- Step 3: Animate Your Image with Veo ---
    st.header("Step 3: Animate Your Image with Veo")
    st.markdown("Provide a **separate prompt** to guide the video generation. Describe the motion, actions, and camera movements you envision.")

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
            value=st.session_state.veo_user_prompt,
            help="Describe what should happen in the video, including character actions or camera movements (e.g., 'camera pans left slowly', 'a car drives by')."
        )
        st.session_state.veo_user_prompt = veo_user_prompt_current_value

        if st.button("🚀 Refine Video Prompt with Gemini", use_container_width=True, type="primary", key="refine_video_prompt_button"):
            if st.session_state.veo_user_prompt:
                with st.spinner("Gemini is refining your video prompt..."):
                    refined_veo_text = refine_prompt_with_gemini(st.session_state.veo_user_prompt, for_video=True)
                    if refined_veo_text:
                        st.session_state.veo_refined_prompt = refined_veo_text
                        st.session_state.veo_user_prompt = refined_veo_text
                st.session_state.video_bytes = None
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

    final_veo_prompt_image_to_video = (
        st.session_state.veo_refined_prompt
        or st.session_state.veo_user_prompt
    )

    st.markdown("---")

    if st.button("🎬 Generate Video from Image", use_container_width=True, 
                 disabled=not (image_for_veo and final_veo_prompt_image_to_video), key="generate_video_button"):
        if not image_for_veo:
            st.warning("Please generate or upload an image in Step 2 first.")
        elif not final_veo_prompt_image_to_video:
            st.warning("Please provide a prompt for video generation in Step 3.")
        else:
            with st.spinner("Veo is animating your image... This can take several minutes."):
                st.session_state.video_bytes = generate_video_with_veo("image", image_for_veo, final_veo_prompt_image_to_video)

    if st.session_state.video_bytes:
        st.subheader("Your Generated Video")
        st.video(st.session_state.video_bytes)
    else:
        st.info("Your generated video will appear here after Veo finishes processing.")

with tab_text_to_video:
    st.header("Text to Video Generation with Veo")
    st.markdown("Describe the entire video you envision. Gemini can help you refine your prompt for better results.")

    col1_text_vid_prompt, col2_text_vid_prompt = st.columns(2)

    with col1_text_vid_prompt:
        text_to_video_user_prompt_current_value = st.text_area(
            "Enter your idea for the video:", 
            height=150, 
            key="text_to_video_user_prompt_input",
            value=st.session_state.text_to_video_user_prompt,
            help="Describe the entire video, including characters, actions, environment, and camera movements."
        )
        st.session_state.text_to_video_user_prompt = text_to_video_user_prompt_current_value

        if st.button("🚀 Refine Video Prompt with Gemini", use_container_width=True, type="primary", key="refine_text_video_prompt_button"):
            if st.session_state.text_to_video_user_prompt:
                with st.spinner("Gemini is refining your video prompt..."):
                    refined_text_to_video_prompt = refine_prompt_with_gemini(st.session_state.text_to_video_user_prompt, for_video=True)
                    if refined_text_to_video_prompt:
                        st.session_state.text_to_video_refined_prompt = refined_text_to_video_prompt
                        st.session_state.text_to_video_user_prompt = refined_text_to_video_prompt
                st.session_state.video_bytes = None
            else:
                st.warning("Please enter an initial idea for the video to refine.")

    with col2_text_vid_prompt:
        if st.session_state.text_to_video_refined_prompt:
            st.markdown("**Gemini's Refined Video Prompt:**")
            st.info(st.session_state.text_to_video_refined_prompt)
        elif st.session_state.text_to_video_user_prompt:
            st.markdown("**Your Video Prompt:**")
            st.info(st.session_state.text_to_video_user_prompt)
        else:
            st.info("Your refined video prompt will appear here.")
    
    final_text_to_video_prompt = (
        st.session_state.text_to_video_refined_prompt
        or st.session_state.text_to_video_user_prompt
    )

    st.markdown("---")

    if st.button("🎬 Generate Video from Text", use_container_width=True, 
                 disabled=not final_text_to_video_prompt, key="generate_text_video_button"):
        if not final_text_to_video_prompt:
            st.warning("Please provide a prompt for video generation.")
        else:
            with st.spinner("Veo is generating your video from text... This can take several minutes."):
                st.session_state.video_bytes = generate_video_with_veo("text", None, final_text_to_video_prompt)

    if st.session_state.video_bytes:
        st.subheader("Your Generated Video")
        st.video(st.session_state.video_bytes)
    else:
        st.info("Your generated video will appear here after Veo finishes processing.")