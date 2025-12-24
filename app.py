import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
from fpdf import FPDF
import io
import random

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Smart Vehicle Emission Monitoring",
    layout="wide"
)

# ---------------------------
# Theme Styling
# ---------------------------
st.markdown("""
<style>
body { background-color:#0b1220; color:white; }
.card { background:#111827; padding:25px; border-radius:16px; margin-bottom:20px }
.metric { font-size:32px; font-weight:bold }
.title { font-size:48px; font-weight:800 }
.subtitle { font-size:18px; opacity:0.8 }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="card">
  <div class="title">Smart Vehicle Emission Monitoring</div>
  <div class="subtitle">
    AI-powered smoke detection & automated pollution enforcement system
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Detection", "e-Challan", "Dashboard", "About"]
)

camera = st.sidebar.selectbox(
    "Camera Location",
    ["Delhi-CP", "Mumbai-BKC", "Bengaluru-ORR", "Chennai-OMR"]
)

# ---------------------------
# Demo Logic (Cloud Safe)
# ---------------------------
def detect_smoke_demo():
    score = round(random.uniform(0.15, 0.55), 2)
    status = "Excessive Smoke" if score >= 0.30 else "Normal Emission"
    confidence = random.randint(60, 95)
    return score, status, confidence

def detect_plate_demo():
    return random.choice([
        "MH20 DV2363",
        "DL8CAF5031",
        "KA01AB1234",
        "TN09CB4455"
    ])

def detect_vehicle_type_demo():
    return random.choice(["Car", "Bus", "Truck", "Two-Wheeler"])

# ---------------------------
# DETECTION PAGE
# ---------------------------
if page == "Detection":
    uploaded = st.file_uploader("Upload Vehicle Image", type=["jpg","png","jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_column_width=True)

        score, status, confidence = detect_smoke_demo()
        plate = detect_plate_demo()
        vehicle = detect_vehicle_type_demo()

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### Smoke Score: `{score}`")
        st.markdown(f"### Status: `{status}`")
        st.markdown(f"### Confidence: `{confidence}%`")
        st.markdown("</div>", unsafe_allow_html=True)

        if status == "Excessive Smoke":
            st.error("🚨 Polluting Vehicle Detected")

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**Number Plate:** `{plate}`")
            st.markdown(f"**Vehicle Type:** `{vehicle}`")
            st.markdown(f"**Camera:** `{camera}`")
            st.markdown(f"**Time:** `{datetime.now()}`")
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# E-CHALLAN PAGE
# ---------------------------
elif page == "e-Challan":
    st.subheader("Auto-Generated e-Challan")

    if st.button("Generate e-Challan"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial","B",16)
        pdf.cell(0,10,"Government of India – e-Challan",ln=1)

        pdf.set_font("Arial","",12)
        pdf.multi_cell(0,8,f"""
Vehicle Number: {detect_plate_demo()}
Vehicle Type: {detect_vehicle_type_demo()}
Violation: Excessive Smoke Emission
Camera: {camera}
Date & Time: {datetime.now()}
Penalty: ₹5000
        """)

        buffer = io.BytesIO()
        pdf.output(buffer)
        buffer.seek(0)

        st.download_button(
            "Download e-Challan PDF",
            buffer,
            file_name="e_challan.pdf",
            mime="application/pdf"
        )

# ---------------------------
# DASHBOARD
# ---------------------------
elif page == "Dashboard":
    data = {
        "Camera": ["Delhi-CP","Mumbai-BKC","Bengaluru-ORR","Chennai-OMR"],
        "Violations": [34, 21, 18, 27]
    }
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index("Camera"))

# ---------------------------
# ABOUT
# ---------------------------
else:
    st.markdown("""
    ### About This Project
    - Cloud-safe AI prototype
    - Designed for hackathon demos
    - OCR & YOLO run on edge servers in production
    - Scalable, compliant & enforcement-ready
    """)
