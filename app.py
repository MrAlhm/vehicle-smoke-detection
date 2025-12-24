import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import uuid
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go

# ==============================
# OPTIONAL YOLO LOADING (SAFE)
# ==============================
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")  # small model
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Smart Vehicle Emission Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# BACKGROUND (POLLUTION RACE CAR)
# ==============================
st.markdown("""
<style>
.stApp {
    background-image:
        linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
        url("https://images.unsplash.com/photo-1542362567-b07e54358753");
    background-size: cover;
    background-attachment: fixed;
}
.fade-in {
    animation: fadeIn 1.2s ease-in;
}
@keyframes fadeIn {
    from {opacity:0; transform: translateY(10px);}
    to {opacity:1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SESSION STATE
# ==============================
if "admin" not in st.session_state:
    st.session_state.admin = False

if "violations" not in st.session_state:
    st.session_state.violations = []

# ==============================
# SIDEBAR NAVIGATION
# ==============================
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Modules",
    [
        "Detection",
        "CCTV Analysis",
        "Violations",
        "e-Challan",
        "Analytics",
        "Admin",
        "About"
    ]
)

camera = st.sidebar.selectbox(
    "Camera Location",
    ["Delhi-CP", "Mumbai-Andheri", "Bengaluru-SilkBoard", "Hyderabad-Hitech"]
)

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class='fade-in' style='padding:25px;border-radius:18px;
background:linear-gradient(135deg,#232526,#414345)'>
<h1 style='color:white;'>Smart Vehicle Emission Monitoring</h1>
<p style='color:white;font-size:16px;'>
AI-powered smoke detection & enforcement platform
</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# SMOKE DETECTION (HEURISTIC)
# ==============================
def detect_smoke(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = (s < 60) & (v > 150)
    score = np.sum(mask) / mask.size
    severity = "High" if score > 0.3 else "Medium" if score > 0.18 else "Low"
    confidence = min(95, int(score * 200))
    return score, severity, confidence

# ==============================
# DETECTION
# ==============================
if menu == "Detection":
    file = st.file_uploader("Upload Vehicle Image", ["jpg","png","jpeg"])

    if file:
        img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)
        st.image(img, channels="BGR")

        score, severity, conf = detect_smoke(img)

        col1,col2,col3 = st.columns(3)
        col1.metric("Smoke Score", f"{score:.2f}")
        col2.metric("Severity", severity)
        col3.metric("Confidence", f"{conf}%")

        if severity == "High":
            st.error("Polluting Vehicle Detected")

            violation = {
                "ID": str(uuid.uuid4())[:8],
                "Camera": camera,
                "Severity": severity,
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.violations.append(violation)

# ==============================
# CCTV ANALYSIS (YOLO)
# ==============================
elif menu == "CCTV Analysis":
    st.info("YOLO Enabled" if YOLO_AVAILABLE else "YOLO disabled (demo mode)")

    if YOLO_AVAILABLE:
        st.success("YOLOv8 Loaded Successfully")
        st.code("Model: yolov8n.pt")
    else:
        st.warning("YOLO not available on this environment")

# ==============================
# VIOLATIONS
# ==============================
elif menu == "Violations":
    if st.session_state.violations:
        df = pd.DataFrame(st.session_state.violations)
        st.dataframe(df)
    else:
        st.info("No violations yet")

# ==============================
# E-CHALLAN
# ==============================
elif menu == "e-Challan":
    if st.session_state.violations:
        last = st.session_state.violations[-1]
        st.write(last)

        if st.button("Download e-Challan"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for k,v in last.items():
                pdf.cell(0,10,f"{k}: {v}",ln=True)
            fname = "e_challan.pdf"
            pdf.output(fname)
            with open(fname,"rb") as f:
                st.download_button("Download PDF", f, fname)
    else:
        st.info("No violation to generate challan")

# ==============================
# ANALYTICS
# ==============================
elif menu == "Analytics":
    if st.session_state.violations:
        df = pd.DataFrame(st.session_state.violations)
        fig = px.histogram(df, x="Camera", color="Severity")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Analytics will appear after detections")

# ==============================
# ADMIN LOGIN
# ==============================
elif menu == "Admin":
    if not st.session_state.admin:
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if user == "admin" and pwd == "admin123":
                st.session_state.admin = True
                st.success("Admin logged in")
            else:
                st.error("Invalid credentials")
    else:
        st.success("Admin Panel")
        st.metric("Total Violations", len(st.session_state.violations))
        if st.button("Logout"):
            st.session_state.admin = False

# ==============================
# ABOUT
# ==============================
else:
    st.write("""
This system demonstrates a scalable AI-based vehicle emission monitoring platform.

Prototype:
• Streamlit UI
• Heuristic smoke detection
• OCR-ready pipeline

Deployment:
• YOLOv8 vehicle & plate detection
• Live CCTV RTSP streams
• Government enforcement integration
""")
