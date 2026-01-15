import gradio as gr
import subprocess
import os
import sys
import glob
from pydub import AudioSegment # For audio conversion

from src.utils.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)

# --- Configuration ---
# Path to your Python inference script
INFERENCE_SCRIPT_PATH = "src/models/inference/moda_test.py"

def convert_audio_to_wav(audio_path):
    """
    Converts an audio file to WAV if necessary and returns the new path.
    """
    try:
        # Determine the new filename with the .wav extension
        path_without_ext, _ = os.path.splitext(audio_path)
        wav_path = f"{path_without_ext}.wav"

        # Load the audio file (MP3, etc.) and export it as WAV
        logger.info(f"Converting audio: {audio_path} -> {wav_path}")
        audio = AudioSegment.from_file(audio_path)
        audio.export(wav_path, format="wav")
        logger.debug(f"Audio conversion completed successfully")
        
        return wav_path
    except Exception as e:
        logger.error(f"Failed to convert audio file to WAV: {e}")
        raise gr.Error(f"Error converting the audio file to WAV: {e}")


def generate_video(image_path, audio_path):
    """
    Executes the inference script in the background and returns the path to the video.
    """
    logger.info("Starting video generation")
    
    # Check that both files have been provided
    if image_path is None or audio_path is None:
        logger.error("Missing required inputs: image or audio file not provided")
        raise gr.Error("Error: You must provide both an image AND an audio file.")

    processed_audio_path = audio_path
    try:
        # --- Audio Conversion Step ---
        # Check if the file is not already a WAV
        if not audio_path.lower().endswith('.wav'):
            processed_audio_path = convert_audio_to_wav(audio_path)
        else:
            logger.debug("Audio file is already in WAV format, skipping conversion")

        # Prepare the command to be executed with the potentially converted audio file
        command = [
            sys.executable,
            INFERENCE_SCRIPT_PATH,
            "--image_path", image_path,
            "--audio_path", processed_audio_path, # Use the path of the WAV file
        ]

        logger.info(f"Executing inference script: {INFERENCE_SCRIPT_PATH}")
        logger.debug(f"Full command: {' '.join(command)}")

        # Create a copy of the current environment and add the variable
        # to force UTF-8, which fixes encoding errors on Windows.
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # Execute the command and wait for it to finish.
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=env # Use the modified environment
        )

        # Display the script's output information (useful for debugging)
        logger.debug(f"Script STDOUT: {process.stdout}")
        if process.stderr:
            logger.debug(f"Script STDERR: {process.stderr}")

        # Search for the most recent .mp4 video file in the entire project
        list_of_files = glob.glob('**/*.mp4', recursive=True)
        if not list_of_files:
            logger.error("No .mp4 video file found after script execution")
            raise gr.Error("The script finished, but no .mp4 video file was found. Please check if the script generated a video correctly.")

        latest_file = max(list_of_files, key=os.path.getctime)
        logger.info(f"Video generation completed successfully: {latest_file}")

        # If everything went well, return the path of the most recent video
        return latest_file

    except subprocess.CalledProcessError as e:
        # If the script returns an error, display it in the interface
        logger.error(f"Inference script failed with error: {e.stderr}")
        raise gr.Error(f"The inference script failed. Error:\n\n{e.stderr}")
    except Exception as e:
        # Handle other possible errors
        logger.exception(f"Unexpected error during video generation: {e}")
        raise gr.Error(f"An unexpected error occurred: {str(e)}")
    finally:
        # Cleanup: delete the temporary WAV file if it was created
        if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
            logger.debug(f"Cleaning up temporary file: {processed_audio_path}")
            os.remove(processed_audio_path)


# --- Gradio Interface ---
# Using gr.Blocks for a custom layout
with gr.Blocks(title="MODA Video Generator") as demo:
    gr.Markdown(
        """
        # 🎬 AI Video Generator (MODA)
        Provide a source image and an audio file to animate the face.
        """
    )

    # Organize the interface into two columns
    with gr.Row():
        with gr.Column(scale=1):
            # --- LEFT COLUMN: INPUTS ---
            gr.Markdown("### 1. Input Files")
            image_input = gr.Image(type="filepath", label="Source Image (.jpg, .png)")
            audio_input = gr.Audio(type="filepath", label="Audio File (.wav, .mp3)")
            
            submit_button = gr.Button("▶️ Start Generation", variant="primary")

        with gr.Column(scale=1):
            # --- RIGHT COLUMN: OUTPUT ---
            gr.Markdown("### 2. Result")
            video_output = gr.Video(label="Generated Video")

    # Connect the button click to the generation function
    submit_button.click(
        fn=generate_video,
        inputs=[image_input, audio_input],
        outputs=[video_output]
    )


# --- App Launch ---
if __name__ == "__main__":
    logger.info("Starting MoDA Video Generator application")
    # `share=True` is necessary for certain environments (e.g., Pinokio, Google Colab)
    # Note: theme moved to launch() in Gradio 6.0
    demo.launch(share=True, theme=gr.themes.Glass())