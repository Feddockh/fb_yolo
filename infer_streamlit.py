"""Streamlit app for running YOLO inference on webcam or uploaded images.

Usage:
  pip install streamlit ultralytics opencv-python
  streamlit run infer_streamlit.py

This app supports:
 - Uploading an image to run inference on
 - Using the browser camera (via streamlit's camera_input) for snapshots
 - A basic "live" mode that captures frames via OpenCV (if available) and updates the image
"""
from pathlib import Path
import time
import streamlit as st
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from ultralytics import YOLO

MODEL_CACHE = {}


@st.cache_resource
def load_model(weights: str):
    model = YOLO(weights)
    return model


def annotate_frame(frame_bgr, model, conf=0.25, iou=0.45):
    # run inference
    results = model(frame_bgr, conf=conf, iou=iou)
    res = results[0]
    annotated = res.plot()  # BGR numpy
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB) if cv2 is not None else annotated
    return annotated_rgb


def main():
    st.title("YOLO Streamlit Webcam/Upload Inference")

    st.sidebar.header("Model & Source")
    weights = st.sidebar.text_input("Weights path", "runs/train/yolov8_large_rivendale_v6_k_fold/yolov8_large_rivendale_v6_k_fold1/weights/best.pt")
    conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25)
    iou = st.sidebar.slider("NMS IoU", 0.0, 1.0, 0.45)
    mode = st.sidebar.selectbox("Mode", ["Upload image", "Camera snapshot", "Live (OpenCV)"])

    model = None
    if weights:
        try:
            model = load_model(weights)
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")

    if mode == "Upload image":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded and model is not None:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated = annotate_frame(img, model, conf=conf, iou=iou)
            st.image(annotated, channels="RGB")

    elif mode == "Camera snapshot":
        cam = st.camera_input("Take a photo")
        if cam is not None and model is not None:
            file_bytes = np.frombuffer(cam.getvalue(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated = annotate_frame(img, model, conf=conf, iou=iou)
            st.image(annotated, channels="RGB")

    else:  # Live (OpenCV)
        if cv2 is None:
            st.error("OpenCV is not available in this environment. Live mode requires OpenCV.")
            return

        source = st.sidebar.text_input("OpenCV source", "0")
        run = st.button("Start live")
        stop = st.button("Stop")
        placeholder = st.empty()

        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            st.error(f"Unable to open source: {source}")
            return

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated = annotate_frame(frame, model, conf=conf, iou=iou)
                placeholder.image(annotated, channels="RGB")
                # small sleep to yield control and avoid blocking Streamlit
                time.sleep(0.03)
                # check for stop via button is not trivial; refresh the page to stop
        finally:
            cap.release()


if __name__ == '__main__':
    main()
