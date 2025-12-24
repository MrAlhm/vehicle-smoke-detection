import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import plotly.graph_objects as go

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Smart City Vehicle Smoke Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# BACKGROUND IMAGE
# -------------------------------
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image:
                linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
                url("https://images.unsplash.com/photo-1509395176047-4a66953fd231");
            background-size: cover;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("Navigation")

mode = st.sidebar.radio(
    "Select Module",
    [
        "Detection",
        "CCTV Video",
        "Dashboard",
        "About"
    ]
)

camera_location = st.sidebar.selectbox(
    "Camera Location",
    ["Delhi-Cam-01", "Mumbai-Cam-07", "Bengaluru-Cam-12", "Chennai-Cam-04"]
)

# -------------------------------
# HEADER / LANDING SECTION
# -------------------------------
st.markdown(
    """
    <div style="padding:25px;border-radius:20px;
    background:linear-gradient(135deg,#1e3c72,#2a5298)">
    <h1 style="color:white;">Smart City Vehicle Smoke Monitoring</h1>
    <p style="color:white;font-size:18px;">
    AI-powered detection of vehicular pollution using images and CCTV footage
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# -------------------------------
# SMOKE DETECTION LOGIC (SAFE HEURISTIC)
# -------------------------------
def detect_smoke(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    smoke_mask = (s < 60) & (v > 150)
    smoke_score = np.sum(smoke_mask) / smoke_mask.size

    if smoke_score > 0.30:
        severity = "High"
    elif smoke_score > 0.18:
        severity = "Medium"
    else:
        severity = "Low"

    confidence = int(min(95, smoke_score * 200))

    return smoke_score, severity, confidence

# -------------------------------
# AI CONFIDENCE GRAPH
# -------------------------------
def confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#00ffcc"},
            "steps": [
                {"range": [0, 40], "color": "#2ecc71"},
                {"range": [40, 70], "color": "#f1c40f"},
                {"range": [70, 100], "color": "#e74c3c"}
            ],
        },
        title={"text": "AI Detection Confidence (%)"}
    ))
    fig.update_layout(height=250, margin=dict(t=30, b=10))
    return fig

# -------------------------------
# DETECTION MODULE (IMAGE)
# -------------------------------
if mode == "Detection":
    st.subheader("Vehicle Image Analysis")

    uploaded_image = st.file_uploader(
        "Upload Vehicle Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, use_column_width=True)

        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        score, severity, confidence = detect_smoke(image_bgr)

        col1, col2, col3 = st.columns(3)
        col1.metric("Smoke Score", f"{score:.2f}")
        col2.metric("Severity", severity)
        col3.metric("Confidence", f"{confidence}%")

        st.plotly_chart(confidence_gauge(confidence), use_container_width=True)

        if severity == "High":
            st.error("Polluting Vehicle Detected")
            st.info("Number Plate Detection: (YOLO-based – demo placeholder)")
        else:
            st.success("Emission Within Permissible Limit")

        st.caption(f"Camera: {camera_location}")
        st.caption(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# -------------------------------
# CCTV VIDEO MODULE
# -------------------------------
elif mode == "CCTV Video":
    st.subheader("CCTV Video Analysis")

    uploaded_video = st.file_uploader(
        "Upload CCTV Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        st.video(uploaded_video)
        st.info(
            "Video uploaded successfully.\n"
            "Frame-by-frame smoke analysis & vehicle tracking enabled in full deployment."
        )

# -------------------------------
# DASHBOARD
# -------------------------------
elif mode == "Dashboard":
    st.subheader("Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Vehicles Checked", 1284)
        st.metric("Violations Detected", 214)

    with col2:
        st.metric("High-Risk Cameras", 6)
        st.metric("Avg AI Confidence", "78%")

    st.plotly_chart(confidence_gauge(78), use_container_width=True)

# -------------------------------
# ABOUT
# -------------------------------
elif mode == "About":
    st.subheader("About This System")

    st.write(
        """
        This prototype demonstrates an AI-powered system for detecting
        excessive vehicular smoke using traffic camera images and CCTV footage.

        It is designed for:
        • Smart Cities  
        • Traffic Enforcement  
        • Pollution Control Boards  

        Built as a hackathon-ready, scalable proof-of-concept.
        """
    )

# -------------------------------
# FOOTER
# -------------------------------
st.markdown(
    "<hr style='border:0.5px solid #444;'>"
    "<p style='text-align:center;color:gray;'>Smart City AI • Hackathon Prototype</p>",
    unsafe_allow_html=True
)
