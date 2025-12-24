import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import easyocr
from fpdf import FPDF
import plotly.graph_objects as go
import uuid

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Smart City Vehicle Smoke Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# BACKGROUND (ENVIRONMENT THEME)
# -------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image:
            linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
            url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
        background-size: cover;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("Navigation")

mode = st.sidebar.radio(
    "Module",
    ["Detection", "e-Challan", "Dashboard", "About"]
)

camera_location = st.sidebar.selectbox(
    "Camera Location",
    ["Delhi-01", "Mumbai-05", "Bengaluru-03", "Chennai-02"]
)

# -------------------------------
# HEADER
# -------------------------------
st.markdown(
    """
    <div style="padding:25px;border-radius:20px;
    background:linear-gradient(135deg,#0f2027,#203a43,#2c5364)">
    <h1 style="color:white;">Smart City Vehicle Smoke Monitoring</h1>
    <p style="color:white;font-size:17px;">
    AI-powered pollution detection using traffic cameras
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# -------------------------------
# OCR READER
# -------------------------------
reader = easyocr.Reader(['en'], gpu=False)

# -------------------------------
# SMOKE DETECTION
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
# NUMBER PLATE OCR (REAL)
# -------------------------------
def detect_number_plate(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)

    for res in results:
        text = res[1]
        if len(text) >= 6:
            return text.upper()

    return "Not Readable"

# -------------------------------
# VEHICLE TYPE (RULE-BASED)
# -------------------------------
def estimate_vehicle_type(image):
    h, w, _ = image.shape
    if w > 900:
        return "Truck / Bus"
    elif w > 600:
        return "Car"
    else:
        return "Two-Wheeler"

# -------------------------------
# CONFIDENCE GAUGE
# -------------------------------
def confidence_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": "AI Confidence"},
        gauge={"axis": {"range": [0, 100]}}
    ))
    fig.update_layout(height=250)
    return fig

# -------------------------------
# PDF GENERATOR (REAL DOWNLOAD)
# -------------------------------
def generate_challan(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "AUTO-GENERATED E-CHALLAN", ln=True, align="C")
    pdf.ln(10)

    for k, v in data.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    file_name = f"e_challan_{data['Violation ID']}.pdf"
    pdf.output(file_name)
    return file_name

# -------------------------------
# DETECTION MODULE
# -------------------------------
if mode == "Detection":
    uploaded_image = st.file_uploader(
        "Upload Vehicle Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, use_column_width=True)

        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        score, severity, confidence = detect_smoke(img_bgr)
        plate = detect_number_plate(img_bgr)
        vehicle_type = estimate_vehicle_type(img_np)

        st.subheader("Detection Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Smoke Score", f"{score:.2f}")
        c2.metric("Severity", severity)
        c3.metric("Vehicle Type", vehicle_type)

        st.plotly_chart(confidence_gauge(confidence), use_container_width=True)

        if severity == "High":
            st.error("Polluting Vehicle Detected")
            st.info(f"Detected Number Plate: {plate}")
            st.session_state["last_violation"] = {
                "Violation ID": str(uuid.uuid4())[:8],
                "Plate": plate,
                "Vehicle Type": vehicle_type,
                "Severity": severity,
                "Camera": camera_location,
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            st.success("Emission Within Permissible Limit")

# -------------------------------
# E-CHALLAN MODULE
# -------------------------------
elif mode == "e-Challan":
    if "last_violation" in st.session_state:
        data = st.session_state["last_violation"]
        st.subheader("Auto-Generated e-Challan")

        for k, v in data.items():
            st.write(f"**{k}:** {v}")

        if st.button("Download e-Challan PDF"):
            file = generate_challan(data)
            with open(file, "rb") as f:
                st.download_button(
                    "Click to Download",
                    f,
                    file_name=file,
                    mime="application/pdf"
                )
    else:
        st.warning("No violation detected yet.")

# -------------------------------
# DASHBOARD
# -------------------------------
elif mode == "Dashboard":
    st.metric("Total Vehicles Monitored", 1342)
    st.metric("Violations Detected", 231)
    st.metric("High Risk Cameras", 5)

# -------------------------------
# ABOUT
# -------------------------------
elif mode == "About":
    st.write(
        """
        This system demonstrates an AI-powered solution for
        detecting vehicular smoke using traffic camera imagery.

        **Prototype Uses:**  
        • Image-based smoke analysis  
        • OCR-based plate reading  

        **Real Deployment Uses:**  
        • YOLO vehicle + plate detection  
        • CCTV streaming  
        • Automated enforcement
        """
    )
