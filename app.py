import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
from fpdf import FPDF
import easyocr
import re
import uuid

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Smart Vehicle Emission Monitoring",
    layout="wide"
)

# ======================================
# OCR INIT
# ======================================
reader = easyocr.Reader(['en'], gpu=False)

# ======================================
# SESSION STORAGE
# ======================================
if "violations" not in st.session_state:
    st.session_state.violations = []

# ======================================
# SIDEBAR
# ======================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Detection", "e-Challan", "Dashboard", "About"]
)

camera_location = st.sidebar.selectbox(
    "Camera Location",
    ["Delhi-CP", "Mumbai-Andheri", "Bengaluru-SilkBoard", "Hyderabad-Hitech"]
)

# ======================================
# HEADER
# ======================================
st.markdown("""
<div style="padding:25px;border-radius:18px;
background:linear-gradient(135deg,#1f2933,#111827)">
<h1 style="color:white;">Smart Vehicle Emission Monitoring</h1>
<p style="color:#d1d5db;font-size:16px;">
AI-powered smoke detection & enforcement platform
</p>
</div>
""", unsafe_allow_html=True)

# ======================================
# CORE FUNCTIONS
# ======================================
def detect_smoke(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    smoke_mask = (s < 60) & (v > 150)
    score = np.sum(smoke_mask) / smoke_mask.size

    if score > 0.35:
        severity = "High"
    elif score > 0.20:
        severity = "Medium"
    else:
        severity = "Low"

    confidence = min(95, int(score * 200))
    return score, severity, confidence


def detect_number_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.018 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate = image[y:y+h, x:x+w]

            result = reader.readtext(plate)
            if result:
                text = result[0][1]
                text = re.sub(r'[^A-Z0-9]', '', text.upper())
                return text, plate

    return "Not Readable", None


def classify_vehicle(image):
    h, w, _ = image.shape
    ratio = w / h
    if ratio > 2.5:
        return "Bus / Truck"
    else:
        return "Car"


def generate_challan(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Government of India", ln=True, align="C")
    pdf.cell(0, 10, "Electronic Traffic Violation Notice", ln=True, align="C")
    pdf.ln(10)

    for k, v in data.items():
        pdf.cell(0, 10, f"{k}: {v}", ln=True)

    filename = f"e_challan_{data['Violation ID']}.pdf"
    pdf.output(filename)
    return filename

# ======================================
# DETECTION PAGE
# ======================================
if page == "Detection":
    uploaded = st.file_uploader(
        "Upload Vehicle Image (CCTV Frame)",
        ["jpg", "jpeg", "png"]
    )

    if uploaded:
        image = Image.open(uploaded)
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        st.image(image, caption="Captured Vehicle Frame", use_column_width=True)

        score, severity, confidence = detect_smoke(img_bgr)
        plate_text, plate_img = detect_number_plate(img_bgr)
        vehicle_type = classify_vehicle(img_bgr)

        st.subheader("Detection Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Smoke Score", f"{score:.2f}")
        c2.metric("Severity", severity)
        c3.metric("Confidence", f"{confidence}%")

        if severity == "High":
            st.error("Polluting Vehicle Detected")

            if plate_img is not None:
                st.image(plate_img, caption="Detected Number Plate", width=300)

            st.info(f"Number Plate: {plate_text}")
            st.info(f"Vehicle Type: {vehicle_type}")

            violation = {
                "Violation ID": str(uuid.uuid4())[:8],
                "Number Plate": plate_text,
                "Vehicle Type": vehicle_type,
                "Camera": camera_location,
                "Severity": severity,
                "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            st.session_state.violations.append(violation)

        else:
            st.success("Emission Within Permissible Limit")

# ======================================
# E-CHALLAN PAGE
# ======================================
elif page == "e-Challan":
    if st.session_state.violations:
        last = st.session_state.violations[-1]

        st.subheader("Auto-Generated e-Challan")
        for k, v in last.items():
            st.write(f"**{k}:** {v}")

        if st.button("Download e-Challan"):
            file = generate_challan(last)
            with open(file, "rb") as f:
                st.download_button("Download PDF", f, file)

    else:
        st.info("No violations available")

# ======================================
# DASHBOARD
# ======================================
elif page == "Dashboard":
    if st.session_state.violations:
        df = pd.DataFrame(st.session_state.violations)
        st.dataframe(df)

        chart = df["Camera"].value_counts()
        st.bar_chart(chart)

    else:
        st.info("No data yet")

# ======================================
# ABOUT
# ======================================
else:
    st.write("""
This prototype demonstrates an AI-assisted system for:
• Detecting excessive vehicular smoke  
• Identifying vehicles via number plate OCR  
• Auto-generating government-style e-challans  

Designed for scalability with CCTV & YOLO-based upgrades.
""")
