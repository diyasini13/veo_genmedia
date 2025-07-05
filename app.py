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
import moviepy as mp # Changed import for clarity and common usage
import auth_token

# --- Configuration ---
# Ensure these environment variables are correctly set for your GCP project
# For local development, consider using Streamlit secrets or directly setting in code as you have for demo
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/diyasini/Desktop/Live Demo page/GenMedia/key-svc-gen-ai.json.json'
# os.environ['GOOGLE_CLOUD_PROJECT'] = 'heroprojectlivedemo'
os.environ['GCP_REGION'] = 'us-central1'
project_id = "heroprojectlivedemo"
gcs_bucket_name = "genmedia_video_hero_package"

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
    
def refine_veo_prompt_with_gemini(user_prompt: str, image_bytes: bytes = None, mime_type: str = "image/png") -> str: # CHANGED: Made image_bytes optional, for general use
    """Uses Gemini on Vertex AI to refine a user's prompt for image or video generation, optionally with an image."""
    model = GenerativeModel("gemini-2.0-flash")
    
    refinement_prompt_text = f"""
    You are an expert prompt engineer for text-to-video models.
    Your task is to refine the following user prompt to make it more descriptive, vivid, and detailed for generating a high-quality, engaging video.
    Focus on adding specific details about camera movements (e.g., pan, zoom, tilt), character actions, environmental changes, and dynamic elements.
    If an image is provided, consider how these elements can interact with or animate the visual content of the image.
    Return only the refined prompt, without any additional text or explanation.

    User Prompt: "{user_prompt}"
    """
    
    contents = [Part.from_text(refinement_prompt_text)]
    if image_bytes:
        contents.append(Part.from_data(image_bytes, mime_type=mime_type))
    
    try:
        response = model.generate_content(contents)
        return response.text.strip()
    except Exception as e:
        st.error(f"An error occurred with Gemini: {e}")
        return ""


def generate_image_with_imagen(prompt: str) -> bytes:
    """Generates an image using Imagen on Vertex AI and returns its bytes."""
    try:
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
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
        # Ensure input_content is bytes for image input
        if not isinstance(input_content, bytes):
            st.error("Input content for image generation must be bytes.")
            return None
            
        unique_image_filename = f"veo_input_image_{uuid.uuid4()}.png"
        input_image_gcs_uri = upload_bytes_to_gcs(gcs_bucket_name, unique_image_filename, input_content, content_type="image/png") # Explicitly set content_type
        
        if not input_image_gcs_uri:
            return None

        # st.info(f"Uploaded input image to GCS: {input_image_gcs_uri}")
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

# NEW FUNCTIONS FOR VIDEO STITCHING
def save_temp_video(video_bytes: bytes, segment_index: int) -> str:
    """Saves video bytes to a temporary file and returns the path."""
    temp_dir = "temp_video_segments"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, f"segment_{segment_index}_{uuid.uuid4()}.mp4")
    try:
        with open(temp_filepath, "wb") as f:
            f.write(video_bytes)
        return temp_filepath
    except Exception as e:
        st.error(f"Error saving temporary video file: {e}")
        return None

def stitch_videos(video_paths: list[str]) -> bytes | None:
    """Stitches multiple video files into one and returns the bytes of the combined video."""
    if not video_paths:
        return None

    try:
        st.info("Stitching video segments together...")
        clips = [mp.VideoFileClip(path) for path in video_paths]
        final_clip = mp.concatenate_videoclips(clips)

        output_filename = f"final_stitched_video_{uuid.uuid4()}.mp4"
        output_filepath = os.path.join("temp_video_segments", output_filename)
        
        # Using preset 'ultrafast' for faster processing in Streamlit demo
        final_clip.write_videofile(output_filepath, codec="libx264", audio_codec="aac", fps=24, preset="ultrafast")

        with open(output_filepath, "rb") as f:
            final_video_bytes = f.read()

        # Clean up temporary files and directory
        for path in video_paths:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        # Only remove directory if it's empty
        if os.path.exists("temp_video_segments") and not os.listdir("temp_video_segments"):
            os.rmdir("temp_video_segments")

        st.success("Videos generated successfully!")
        return final_video_bytes

    except Exception as e:
        st.error(f"Error stitching videos with MoviePy: {e}")
        st.warning("Please ensure FFmpeg is installed and accessible in your environment for MoviePy to function correctly.")
        return None
# END NEW FUNCTIONS

# --- Streamlit App UI ---


def main_app():
    """The main user interface for the GenMedia Studio application."""
    st.title("GenMedia Studio ✨")
    st.markdown("Create stunning visuals and videos with Gemini, Imagen, and Veo.")

    # Initialize session state variables
    if 'refined_prompt' not in st.session_state:
        st.session_state.refined_prompt = ""
    if 'image_bytes' not in st.session_state:
        st.session_state.image_bytes = None
    if 'uploaded_image_bytes' not in st.session_state: # This will only be used by the 'Animate Image' tab now
        st.session_state.uploaded_image_bytes = None
    if 'uploaded_image_mime_type' not in st.session_state: # NEW: To store MIME type for refined prompt
        st.session_state.uploaded_image_mime_type = "image/png"
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
    if 'animate_image_uploaded_bytes' not in st.session_state:
        st.session_state.animate_image_uploaded_bytes = None
    if 'animate_image_uploaded_mime_type' not in st.session_state: # NEW: For animate image's uploaded file type
        st.session_state.animate_image_uploaded_mime_type = "image/png"
    if 'animate_image_prompt_option' not in st.session_state:
        st.session_state.animate_image_prompt_option = "Without prompt" # Initialize to "Without prompt"
    if 'animate_image_user_prompt' not in st.session_state:
        st.session_state.animate_image_user_prompt = ""
    if 'animate_image_refined_prompt' not in st.session_state:
        st.session_state.animate_image_refined_prompt = ""
    if 'animate_image_video_bytes' not in st.session_state:
        st.session_state.animate_image_video_bytes = None
    if 'long_video_segments' not in st.session_state: # NEW: For long video segments (file paths)
        st.session_state.long_video_segments = []
    if 'long_video_final_video' not in st.session_state: # NEW: For the final stitched video bytes
        st.session_state.long_video_final_video = None
    if 'long_video_segment_prompts' not in st.session_state: # NEW: To store prompt strings for segments
        st.session_state.long_video_segment_prompts = [""] # Start with one empty prompt

    # --- NEW SESSION STATE FOR IMAGE TO LONG VIDEO ---
    if 'img_long_video_uploaded_image_bytes' not in st.session_state:
        st.session_state.img_long_video_uploaded_image_bytes = None
    if 'img_long_video_uploaded_image_mime_type' not in st.session_state:
        st.session_state.img_long_video_uploaded_image_mime_type = "image/png"
    if 'img_long_video_segment_prompts' not in st.session_state:
        st.session_state.img_long_video_segment_prompts = [""] # Start with one empty prompt
    if 'img_long_video_generated_segments_paths' not in st.session_state:
        st.session_state.img_long_video_generated_segments_paths = []
    if 'img_long_video_final_video' not in st.session_state:
        st.session_state.img_long_video_final_video = None


    # --- Navigation Bar (Using st.tabs) ---
    tab_image_to_video, tab_text_to_video, tab_animate_image, tab_long_video, tab_img_to_long_video = st.tabs([
        "🖼️ Image to Video Generation", 
        "📝 Text to Video Generation", 
        "✨ Animate Image", 
        "🎬 Long Video Generator (Text-Based)", 
        "📸 Image to Long Videos" # NEW TAB
    ]) 

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

        # --- Step 2: Image Generation ---
        st.header("Step 2: Generate a Base Image")
        st.markdown("This image will serve as the static foundation for your video.")

        prompt_to_use_for_imagen = st.session_state.refined_prompt if st.session_state.refined_prompt else st.session_state.user_entered_prompt

        if st.button("🎨 Generate Image with Imagen", use_container_width=True, disabled=not prompt_to_use_for_imagen, key="generate_image_button"):
            if prompt_to_use_for_imagen:
                with st.spinner("Imagen is generating your image... This may take a moment."):
                    st.session_state.image_bytes = generate_image_with_imagen(prompt_to_use_for_imagen)
                    st.session_state.video_bytes = None
                    st.session_state.veo_user_prompt = ""
                    st.session_state.veo_refined_prompt = ""
            else:
                st.warning("Please provide a prompt (in Step 1) to generate an image.")
        
        if st.session_state.image_bytes:
            st.image(st.session_state.image_bytes, caption="Generated by Imagen", use_container_width=True)
        else:
            st.info("Your generated image will appear here.")

        # --- Step 3: Animate Your Image with Veo ---
        st.header("Step 3: Animate Your Image with Veo")
        
        image_for_veo = None
        if st.session_state.image_bytes:
            image_for_veo = st.session_state.image_bytes
            st.info("Using the **Imagen-generated image** for video creation.")
        else:
            st.warning("Please generate an image in Step 2 to proceed with video generation.")

        # Added the radio button for prompt options
        st.subheader("Animation Options")
        st.session_state.animate_image_prompt_option = st.radio(
            "How do you want to animate the image?",
            ("Without prompt", "With prompt"),
            key="image_to_video_prompt_option_radio" # Unique key for this radio button
        )

        final_veo_prompt_image_to_video = ""

        if st.session_state.animate_image_prompt_option == "With prompt":
            st.markdown("Provide a **separate prompt** to guide the video generation. Describe the motion, actions, and camera movements you envision.")
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
                    if st.session_state.veo_user_prompt and st.session_state.image_bytes: # CHANGED: Check for image_bytes
                        with st.spinner("Gemini is refining your video prompt..."):
                            # CHANGED: Pass image_bytes and MIME type
                            refined_veo_text = refine_veo_prompt_with_gemini(st.session_state.veo_user_prompt, st.session_state.image_bytes, st.session_state.uploaded_image_mime_type)
                            if refined_veo_text:
                                st.session_state.veo_refined_prompt = refined_veo_text
                                st.session_state.veo_user_prompt = refined_veo_text
                        st.session_state.video_bytes = None
                    else: # CHANGED: Updated warning message
                        st.warning("Please enter an initial idea for the video and ensure an image is generated in Step 2 to refine.")

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
        else: # "Without prompt" selected
            final_veo_prompt_image_to_video = "Animate the image." # Default prompt for no explicit user prompt

        st.markdown("---")

        if st.button("🎬 Generate Video from Image", use_container_width=True, 
                    disabled=not (image_for_veo and (st.session_state.animate_image_prompt_option == "Without prompt" or final_veo_prompt_image_to_video)), 
                    key="generate_video_button"):
            if not image_for_veo:
                st.warning("Please generate an image in Step 2 first.")
            elif st.session_state.animate_image_prompt_option == "With prompt" and not final_veo_prompt_image_to_video:
                st.warning("Please provide a prompt for video generation or select 'Without prompt'.")
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

    with tab_animate_image:
        st.header("Animate an Image with Veo")
        st.markdown("Upload an image and choose whether to animate it with or without a descriptive prompt.")

        # --- Upload Image Section ---
        st.subheader("Upload Your Image")
        uploaded_animate_file = st.file_uploader("Choose an image to animate...", type=["jpg", "jpeg", "png"], key="animate_image_uploader")

        if uploaded_animate_file is not None:
            st.session_state.animate_image_uploaded_bytes = uploaded_animate_file.getvalue()
            st.session_state.animate_image_uploaded_mime_type = uploaded_animate_file.type # NEW: Store the MIME type
            st.image(st.session_state.animate_image_uploaded_bytes, caption="Image to Animate", width=300)
            st.success("Image uploaded successfully!")
            st.session_state.animate_image_video_bytes = None # Clear previous video
        else:
            st.info("Upload an image (JPG, JPEG, PNG) to animate.")

        # --- Prompt Options ---
        if st.session_state.animate_image_uploaded_bytes:
            st.subheader("Animation Options")
            st.session_state.animate_image_prompt_option = st.radio(
                "How do you want to animate the image?",
                ("Without prompt", "With prompt"),
                key="animate_prompt_option_radio"
            )

            final_animation_prompt = ""

            if st.session_state.animate_image_prompt_option == "With prompt":
                col1_animate_prompt, col2_animate_prompt = st.columns(2)
                with col1_animate_prompt:
                    animate_image_user_prompt_current_value = st.text_area(
                        "Enter your prompt for animation:",
                        height=150,
                        key="animate_image_user_prompt_input",
                        value=st.session_state.animate_image_user_prompt,
                        help="Describe the desired motion, actions, and camera movements for the animation."
                    )
                    st.session_state.animate_image_user_prompt = animate_image_user_prompt_current_value

                    if st.button("🚀 Refine Animation Prompt with Gemini", use_container_width=True, type="primary", key="refine_animate_prompt_button"):
                        if st.session_state.animate_image_user_prompt and st.session_state.animate_image_uploaded_bytes: # CHANGED: Check for bytes
                            with st.spinner("Gemini is refining your animation prompt..."):
                                # CHANGED: Pass mime_type to the refine function
                                refined_animate_text = refine_veo_prompt_with_gemini(
                                    st.session_state.animate_image_user_prompt, 
                                    st.session_state.animate_image_uploaded_bytes,
                                    st.session_state.animate_image_uploaded_mime_type # NEW: Pass the MIME type
                                )
                                if refined_animate_text:
                                    st.session_state.animate_image_refined_prompt = refined_animate_text
                                    st.session_state.animate_image_user_prompt = refined_animate_text
                            st.session_state.animate_image_video_bytes = None
                        else: # CHANGED: Updated warning message
                            st.warning("Please enter an initial idea for the animation and ensure an image is uploaded to refine.")
                with col2_animate_prompt:
                    if st.session_state.animate_image_refined_prompt:
                        st.markdown("**Gemini's Refined Animation Prompt:**")
                        st.info(st.session_state.animate_image_refined_prompt)
                    elif st.session_state.animate_image_user_prompt:
                        st.markdown("**Your Animation Prompt:**")
                        st.info(st.session_state.animate_image_user_prompt)
                    else:
                        st.info("Your refined animation prompt will appear here.")
                
                final_animation_prompt = (
                    st.session_state.animate_image_refined_prompt
                    or st.session_state.animate_image_user_prompt
                )
            else: # "Without prompt" selected
                final_animation_prompt = "Animate the image." # Default prompt

            st.markdown("---")
            
            if st.button("🎬 Animate Image", use_container_width=True, 
                        disabled=not (st.session_state.animate_image_uploaded_bytes and 
                                    (st.session_state.animate_image_prompt_option == "Without prompt" or final_animation_prompt)), 
                        key="animate_image_button"):
                
                if not st.session_state.animate_image_uploaded_bytes:
                    st.warning("Please upload an image first.")
                elif st.session_state.animate_image_prompt_option == "With prompt" and not final_animation_prompt:
                    st.warning("Please provide a prompt for animation or select 'Without prompt'.")
                else:
                    prompt_to_use = final_animation_prompt if st.session_state.animate_image_prompt_option == "With prompt" else "Animate the image." # Default prompt if none provided
                    with st.spinner("Veo is animating your image... This can take several minutes."):
                        st.session_state.animate_image_video_bytes = generate_video_with_veo(
                            "image", st.session_state.animate_image_uploaded_bytes, prompt_to_use
                        )

            if st.session_state.animate_image_video_bytes:
                st.subheader("Your Animated Video")
                st.video(st.session_state.animate_image_video_bytes)
            else:
                st.info("Your animated video will appear here after Veo finishes processing.")

    # Existing Long Video Generator (Text-Based)
    with tab_long_video:
        st.header("Generate Long Videos (Text-Based)")
        st.markdown("Since Veo generates videos up to 8 seconds, you can create longer videos by defining multiple segments and stitching them together.")

        # Dynamic prompt input fields for segments
        st.subheader("Define Video Segments")
        
        for i, prompt_value in enumerate(st.session_state.long_video_segment_prompts):
            st.markdown(f"**Segment {i+1}**")
            col_prompt, col_refine = st.columns([0.7, 0.3])
            
            with col_prompt:
                segment_prompt_current_value = st.text_area(
                    f"Prompt for Segment {i+1}:", 
                    height=100, 
                    key=f"long_video_segment_prompt_{i}",
                    value=prompt_value,
                    help="Describe the scene and action for this video segment. Be mindful of continuity if you're stitching."
                )
                st.session_state.long_video_segment_prompts[i] = segment_prompt_current_value
                
            with col_refine:
                st.markdown("<br>", unsafe_allow_html=True) # Spacer for alignment
                if st.button(f"✨ Refine Segment {i+1} Prompt", key=f"refine_long_segment_prompt_{i}", use_container_width=True):
                    if segment_prompt_current_value:
                        with st.spinner(f"Refining prompt for Segment {i+1}..."):
                            refined_segment_prompt = refine_prompt_with_gemini(segment_prompt_current_value, for_video=True)
                            if refined_segment_prompt:
                                st.session_state.long_video_segment_prompts[i] = refined_segment_prompt
                                st.rerun() # Rerun to update the text area with refined prompt
                    else:
                        st.warning("Please enter a prompt for this segment to refine.")

        col_add_remove = st.columns(2)
        with col_add_remove[0]:
            if st.button("➕ Add Another Segment", use_container_width=True):
                st.session_state.long_video_segment_prompts.append("")
                st.session_state.long_video_final_video = None # Clear previous
                st.session_state.long_video_segments = [] # Clear generated segments
                st.rerun() # Rerun to show new input field
        with col_add_remove[1]:
            if len(st.session_state.long_video_segment_prompts) > 1:
                if st.button("➖ Remove Last Segment", use_container_width=True):
                    st.session_state.long_video_segment_prompts.pop()
                    st.session_state.long_video_final_video = None
                    st.session_state.long_video_segments = []
                    st.rerun()


        st.markdown("---")
        st.subheader("Generate and Stitch")

        # Generate Segments Button
        if st.button("🚀 Generate All Video Segments", use_container_width=True, type="primary", key="generate_all_segments_button"):
            all_prompts_valid = True
            for prompt in st.session_state.long_video_segment_prompts:
                if not prompt.strip():
                    all_prompts_valid = False
                    st.error("Please ensure all segment prompts are filled in.")
                    break
            
            if all_prompts_valid:
                st.session_state.long_video_segments = [] # Reset generated segments
                st.session_state.long_video_final_video = None # Reset final video
                
                total_segments = len(st.session_state.long_video_segment_prompts)
                segment_progress_bar = st.progress(0)
                segment_status_text = st.empty()

                for i, prompt in enumerate(st.session_state.long_video_segment_prompts):
                    segment_status_text.text(f"Generating segment {i+1} of {total_segments}...")
                    with st.spinner(f"Veo is generating segment {i+1}..."):
                        segment_video_bytes = generate_video_with_veo("text", None, prompt) # Use "text" input type
                        if segment_video_bytes:
                            temp_filepath = save_temp_video(segment_video_bytes, i)
                            if temp_filepath:
                                st.session_state.long_video_segments.append(temp_filepath)
                        else:
                            st.error(f"Failed to generate segment {i+1}. Aborting stitching.")
                            st.session_state.long_video_segments = [] # Clear any partially generated segments
                            break
                    segment_progress_bar.progress(int(((i + 1) / total_segments) * 100))
                
                if len(st.session_state.long_video_segments) == total_segments and total_segments > 0:
                    st.success("All segments generated successfully!")
                    # Automatically stitch if all segments are generated
                    final_stitched_video_bytes = stitch_videos(st.session_state.long_video_segments)
                    if final_stitched_video_bytes:
                        st.session_state.long_video_final_video = final_stitched_video_bytes
                elif total_segments == 0:
                    st.warning("No segments defined to generate.")
                else:
                    st.error("Failed to generate all segments for stitching.")

        if st.session_state.long_video_final_video:
            st.subheader("Your Long Generated Video")
            st.video(st.session_state.long_video_final_video)
        elif st.session_state.long_video_segments:
            st.info(f"Generated {len(st.session_state.long_video_segments)} segments. Click 'Generate All Video Segments' again if you want to retry stitching.")
            st.warning("If the final video does not appear, ensure FFmpeg is correctly installed and accessible on your system.")
        else:
            st.info("Define your video segments above and click 'Generate All Video Segments' to create your long video.")




    ### **New Tab: Image to Long Videos**


    with tab_img_to_long_video: # This is the new tab
        st.header("Image to Long Videos")
        st.markdown("Upload a base image and define multiple prompt segments to create a long video, with each segment animating the uploaded image.")

        # --- Step 1: Upload Base Image ---
        st.subheader("Step 1: Upload Your Base Image")
        uploaded_img_long_file = st.file_uploader(
            "Choose an image to be the base for all video segments...", 
            type=["jpg", "jpeg", "png"], 
            key="img_long_video_uploader"
        )

        if uploaded_img_long_file is not None:
            st.session_state.img_long_video_uploaded_image_bytes = uploaded_img_long_file.getvalue()
            st.session_state.img_long_video_uploaded_image_mime_type = uploaded_img_long_file.type
            st.image(st.session_state.img_long_video_uploaded_image_bytes, caption="Base Image for Long Video", width=300)
            st.success("Base image uploaded successfully!")
            # Reset segments and final video if a new image is uploaded
            st.session_state.img_long_video_generated_segments_paths = []
            st.session_state.img_long_video_final_video = None
        else:
            st.info("Upload a single image (JPG, JPEG, PNG) that will be animated across all video segments.")

        # --- Step 2: Define Video Segments ---
        st.subheader("Step 2: Define Video Segments")
        st.markdown("For each segment, describe the animation or action you want to see applied to the uploaded image. Consider continuity!")
        
        # Ensure there's at least one prompt input if none exist
        if not st.session_state.img_long_video_segment_prompts:
            st.session_state.img_long_video_segment_prompts.append("")

        for i, prompt_value in enumerate(st.session_state.img_long_video_segment_prompts):
            st.markdown(f"**Segment {i+1}**")
            col_prompt, col_refine = st.columns([0.7, 0.3])
            
            with col_prompt:
                segment_prompt_current_value = st.text_area(
                    f"Prompt for Segment {i+1}:", 
                    height=100, 
                    key=f"img_long_video_segment_prompt_{i}",
                    value=prompt_value,
                    help="Describe the motion, actions, and camera movements for this specific segment, referencing the uploaded image."
                )
                st.session_state.img_long_video_segment_prompts[i] = segment_prompt_current_value
                
            with col_refine:
                st.markdown("<br>", unsafe_allow_html=True) # Spacer for alignment
                if st.button(f"✨ Refine Segment {i+1} Prompt (Gemini)", key=f"refine_img_long_segment_prompt_{i}", use_container_width=True):
                    if segment_prompt_current_value and st.session_state.img_long_video_uploaded_image_bytes:
                        with st.spinner(f"Refining prompt for Segment {i+1}..."):
                            # Pass the uploaded image bytes to Gemini for context
                            refined_segment_prompt = refine_veo_prompt_with_gemini(
                                segment_prompt_current_value, 
                                st.session_state.img_long_video_uploaded_image_bytes, 
                                st.session_state.img_long_video_uploaded_image_mime_type
                            )
                            if refined_segment_prompt:
                                st.session_state.img_long_video_segment_prompts[i] = refined_segment_prompt
                                st.rerun() # Rerun to update the text area with refined prompt
                    else:
                        st.warning("Please enter a prompt for this segment AND upload a base image to refine.")

        col_add_remove_img_long = st.columns(2)
        with col_add_remove_img_long[0]:
            if st.button("➕ Add Another Segment", use_container_width=True, key="add_img_long_segment"):
                st.session_state.img_long_video_segment_prompts.append("")
                st.session_state.img_long_video_final_video = None # Clear previous
                st.session_state.img_long_video_generated_segments_paths = [] # Clear generated segments
                st.rerun() 
        with col_add_remove_img_long[1]:
            if len(st.session_state.img_long_video_segment_prompts) > 1:
                if st.button("➖ Remove Last Segment", use_container_width=True, key="remove_img_long_segment"):
                    st.session_state.img_long_video_segment_prompts.pop()
                    st.session_state.img_long_video_final_video = None
                    st.session_state.img_long_video_generated_segments_paths = []
                    st.rerun()

        st.markdown("---")
        st.subheader("Step 3: Generate and Stitch")

        generate_img_long_disabled = not st.session_state.img_long_video_uploaded_image_bytes or \
                                    any(not p.strip() for p in st.session_state.img_long_video_segment_prompts)

        if st.button("🎬 Generate & Stitch All Videos (Image-Based)", use_container_width=True, type="primary", disabled=generate_img_long_disabled, key="generate_img_long_video_button"):
            if not st.session_state.img_long_video_uploaded_image_bytes:
                st.error("Please upload a base image in Step 1.")
            elif any(not p.strip() for p in st.session_state.img_long_video_segment_prompts):
                st.error("Please ensure all segment prompts are filled in.")
            else:
                st.session_state.img_long_video_generated_segments_paths = [] # Reset generated paths
                st.session_state.img_long_video_final_video = None # Reset final video
                
                total_segments = len(st.session_state.img_long_video_segment_prompts)
                segment_progress_bar = st.progress(0)
                segment_status_text = st.empty()

                for i, prompt in enumerate(st.session_state.img_long_video_segment_prompts):
                    segment_status_text.text(f"Generating segment {i+1} of {total_segments} (Image-Based)...")
                    with st.spinner(f"Veo is generating segment {i+1} for your image..."):
                        # THIS IS THE CRUCIAL PART: Pass the uploaded image bytes for each segment
                        segment_video_bytes = generate_video_with_veo(
                            "image", 
                            st.session_state.img_long_video_uploaded_image_bytes, 
                            prompt
                        )
                        if segment_video_bytes:
                            temp_filepath = save_temp_video(segment_video_bytes, i)
                            if temp_filepath:
                                st.session_state.img_long_video_generated_segments_paths.append(temp_filepath)
                        else:
                            st.error(f"Failed to generate segment {i+1}. Aborting stitching.")
                            st.session_state.img_long_video_generated_segments_paths = [] # Clear any partially generated segments
                            break # Stop further generation if one fails
                    segment_progress_bar.progress(int(((i + 1) / total_segments) * 100))
                
                if len(st.session_state.img_long_video_generated_segments_paths) == total_segments and total_segments > 0:
                    st.success("All image-based segments generated successfully! Now stitching...")
                    final_stitched_video_bytes = stitch_videos(st.session_state.img_long_video_generated_segments_paths)
                    if final_stitched_video_bytes:
                        st.session_state.img_long_video_final_video = final_stitched_video_bytes
                elif total_segments == 0:
                    st.warning("No segments defined to generate.")
                else:
                    st.error("Failed to generate all segments for stitching.")

        if st.session_state.img_long_video_final_video:
            st.subheader("Your Long Image-Based Video")
            st.video(st.session_state.img_long_video_final_video)
        elif st.session_state.img_long_video_generated_segments_paths:
            st.info(f"Generated {len(st.session_state.img_long_video_generated_segments_paths)} segments. Click 'Generate & Stitch All Videos (Image-Based)' again if you want to retry stitching.")
            st.warning("If the final video does not appear, ensure FFmpeg is correctly installed and accessible on your system.")
        else:
            st.info("Upload a base image and define your video segments above, then click 'Generate & Stitch All Videos (Image-Based)' to create your long video.")

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="GenMedia Studio ✨")
    user_info = auth_token.get_authenticated_user()

    if user_info:
        # Display user info and logout button in the sidebar
        with st.sidebar:
            st.image(user_info.get("picture"), width=75)
            st.markdown(f"**Welcome, {user_info.get('name')}!**")
            st.write(f"_{user_info.get('email')}_")
            if st.button("Logout"):
                if 'user_info' in st.session_state:
                    del st.session_state['user_info']
                st.rerun()
        
        main_app()  # Run the main application UI
    else:
        # Show login page
        st.title("Welcome to GenMedia Studio!")
        st.write("Please sign in with your Google account to continue.")
        auth_token.display_login_button()