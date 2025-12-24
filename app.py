import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import easyocr
from fpdf import FPDF
import tempfile
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Smart Vehicle Emission Monitoring",
    layout="wide"
)

# ==============================
# BACKGROUND + STYLES
# ==============================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            rgba(0,0,0,0.75),
            rgba(0,0,0,0.75)
        ),
        url("https://images.unsplash.com/photo-1503376780353-7e6692767b70");
        background-size: cover;
        background-attachment: fixed;
    }
    .card {
        background: rgba(20,20,20,0.9);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# OCR MODEL
# ==============================
reader = easyocr.Reader(['en'], gpu=False)

# ==============================
# SMOKE DETECTION
# ==============================
def detect_smoke(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    smoke_mask = (s < 70) & (v > 140)
    score = np.sum(smoke_mask) / smoke_mask.size

    if score >= 0.30:
        severity = "High"
    elif score >= 0.18:
        severity = "Moderate"
    else:
        severity = "Low"

    return score, severity

# ==============================
# NUMBER PLATE DETECTION
# ==============================
def detect_number_plate(image_bgr):
    result = reader.readtext(image_bgr)
    for r in result:
        text = r[1]
        if len(text) >= 6 and any(char.isdigit() for char in text):
            return text.replace(" ", "")
    return "Not Readable"

# ==============================
# VEHICLE TYPE (SAFE HEURISTIC)
# ==============================
def detect_vehicle_type(image):
    h, w, _ = image.shape
    ratio = w / h

    if ratio > 2.2:
        return "Bus / Truck"
    elif ratio > 1.6:
        return "Car"
    else:
        return "Two-Wheeler"

# ==============================
# E-CHALLAN PDF
# ==============================
def generate_challan(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Government of India", ln=True)
    pdf.cell(0, 10, "Electronic Traffic Violation Challan", ln=True)
    pdf.ln(5)

    for k, v in data.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# ==============================
# SIDEBAR NAV
# ==============================
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Detection", "e-Challan", "About"]
)

# ==============================
# HERO
# ==============================
st.markdown(
    """
    <div class='card'>
        <h1>Smart Vehicle Emission Monitoring</h1>
        <p>AI-powered smoke detection & enforcement platform</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================
# DETECTION SECTION
# ==============================
if section == "Detection":

    uploaded = st.file_uploader(
        "Upload Vehicle Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        img = Image.open(uploaded)
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Captured Frame", use_column_width=True)

        smoke_score, severity = detect_smoke(img_bgr)
        plate = detect_number_plate(img_bgr)
        vtype = detect_vehicle_type(img_np)

        confidence = min(int(smoke_score * 200), 100)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.metric("Smoke Score", f"{smoke_score:.2f}")
            st.metric("Severity", severity)
            st.metric("Detection Confidence", f"{confidence}%")
            st.metric("Vehicle Type", vtype)
            st.metric("Number Plate", plate)
            st.markdown("</div>", unsafe_allow_html=True)

        if severity != "Low":
            st.error("Polluting Vehicle Detected")

        st.session_state["latest"] = {
            "Plate": plate,
            "Vehicle": vtype,
            "Severity": severity,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# ==============================
# E-CHALLAN SECTION
# ==============================
elif section == "e-Challan":

    if "latest" not in st.session_state:
        st.warning("No violation detected yet.")
    else:
        data = st.session_state["latest"]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Auto-Generated e-Challan")

        for k, v in data.items():
            st.write(f"**{k}**: {v}")

        pdf_path = generate_challan(data)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download e-Challan",
                f,
                file_name="e_challan.pdf"
            )
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# ABOUT
# ==============================
else:
    st.markdown(
        """
        <div class='card'>
        <h3>About</h3>
        <p>
        This prototype demonstrates an AI-powered vehicle emission monitoring system.
        It detects excessive smoke, extracts number plates, identifies vehicle type,
        and generates enforceable digital challans.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
