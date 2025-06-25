import streamlit as st
import tempfile
import os
import cv2
from ultralytics import YOLO
import numpy as np

# 🔐 Step 1: Simple password protection
PASSWORD = "1234"  # Change this to whatever you like

st.set_page_config(page_title="Spark Detection", layout="centered")
st.title("⚡ Spark Detection from Video (YOLOv8-OBB)")

# Ask for password
password = st.text_input("Enter app password:", type="password")
if password != PASSWORD:
    st.warning("❌ Incorrect password. Access denied.")
    st.stop()

st.set_page_config(page_title="Spark Detection", layout="centered")
st.title("⚡ Spark Detection from Video (YOLOv8-OBB)")

# Load trained YOLOv8 model
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n-obb.pt")  # Make sure best.pt is in the same folder or repo
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

if model:
    st.subheader("1️⃣ Upload a Video")
    video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if video_file is not None:
        st.video(video_file)

        if st.button("🔍 Detect Sparks"):
            # Save video to a temp file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            spark_count = 0

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Output video with boxes
            output_path = "output_sparks.mp4"
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height)
            )

            pbar = st.progress(0)
            frame_num = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=0.3, iou=0.4)
                annotated_frame = results[0].plot()

                for det in results[0].obb or []:
                    cls_id = int(det.cls)
                    class_name = model.names[cls_id]
                    if "spark" in class_name.lower():
                        spark_count += 1

                out.write(annotated_frame)

                frame_num += 1
                pbar.progress(min(frame_num / total_frames, 1.0))

            cap.release()
            out.release()
            st.success(f"✅ Detection Complete. Total Sparks Detected: **{spark_count}**")

            # Show download link
            st.subheader("2️⃣ Download or View Result Video")
            with open(output_path, "rb") as f:
                st.download_button("💾 Download Result Video", f, file_name="spark_detection.mp4")
                st.video(f)

