import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import os
import subprocess # Import the subprocess module

@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

def process_video(input_path, output_path, model):
    IMG_SIZE = 640
    CONF_THRESHOLD = 0.5

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open pre-processed video file at {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        st.error("Error: Could not open video writer.")
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)
        rendered_frame = results[0].plot()
        out.write(rendered_frame)

    cap.release()
    out.release()

def main():
    st.title("YOLOv11 Object Detection")
    st.write("This app uses a pre-configured model to detect objects in an uploaded video.")

    model_path = "best.pt"
    # ... (model loading code remains the same) ...
    if not os.path.exists(model_path):
        st.error(f"Error: The model file was not found at the path: {model_path}")
        st.info("Please make sure the 'best.pt' file is located in the correct directory.")
        return
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return

    uploaded_video = st.file_uploader("Upload a video to begin detection (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        input_video_path = None
        downscaled_video_path = None
        output_video_path = None
        try:
            tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1])
            input_video_path = tfile_in.name
            tfile_in.write(uploaded_video.read())
            tfile_in.close()

            st.info("Standardizing video with FFmpeg...")
            tfile_downscaled = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            downscaled_video_path = tfile_downscaled.name
            tfile_downscaled.close()
            
            # --- NEW, SMARTER FFmpeg COMMAND ---
            # This command handles both landscape and portrait videos and ensures even dimensions.
            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-i", input_video_path,
                "-vf", "scale='if(gt(a,1),1280,-2)':'if(gt(a,1),-2,1280)'",
                downscaled_video_path
            ]
            
            subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)

            st.info("Processing the standardized video for object detection...")
            tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            output_video_path = tfile_out.name
            tfile_out.close()
            
            process_video(downscaled_video_path, output_video_path, model)

            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                with open(output_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)
                st.download_button(
                    label="Download Processed Video",
                    data=video_bytes,
                    file_name="video_with_detections.mp4",
                    mime="video/mp4"
                )
            else:
                st.error("Processing failed to produce a video file.")
        
        except subprocess.CalledProcessError as e:
            st.error("FFmpeg failed to process the video. This can happen with certain video formats.")
            st.error(f"FFmpeg Error Details: {e.stderr}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

        finally:
            # Cleanup all temporary files
            if input_video_path and os.path.exists(input_video_path):
                os.remove(input_video_path)
            if downscaled_video_path and os.path.exists(downscaled_video_path):
                os.remove(downscaled_video_path)
            if output_video_path and os.path.exists(output_video_path):
                os.remove(output_video_path)

if __name__ == "__main__":
    main()
