import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import easyocr
import tempfile
import matplotlib.pyplot as plt
from fpdf import FPDF

# ================================
# Page Config
# ================================
st.set_page_config(
    page_title="Vehicle Smoke Detection & Enforcement System",
    layout="wide"
)

# ================================
# OCR Model
# ================================
reader = easyocr.Reader(['en'], gpu=False)

# ================================
# Session State
# ================================
if "violations" not in st.session_state:
    st.session_state.violations = []

# ================================
# Smoke Detection
# ================================
def detect_smoke(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    smoke_mask = (s < 60) & (v > 150)
    smoke_score = np.sum(smoke_mask) / smoke_mask.size

    if smoke_score >= 0.30:
        status = "Excessive Smoke"
        severity = "High"
    elif smoke_score >= 0.20:
        status = "Moderate Smoke"
        severity = "Medium"
    else:
        status = "Normal Emission"
        severity = "Low"

    confidence = min(int(smoke_score * 200), 100)
    return smoke_score, status, severity, confidence

# ================================
# Vehicle Type (Rule-based Demo)
# ================================
def detect_vehicle_type(image_bgr):
    h, w, _ = image_bgr.shape
    if w > 900:
        return "Bus / Truck"
    elif w > 600:
        return "Car"
    else:
        return "Two-Wheeler"

# ================================
# Number Plate OCR
# ================================
def detect_number_plate(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate_img = image_bgr[y:y+h, x:x+w]

            text = reader.readtext(plate_img)
            plate_text = text[0][1] if len(text) else "Not Readable"
            return plate_text, plate_img

    return "Not Readable", None

# ================================
# PDF e-Challan Generator
# ================================
def generate_pdf(challan):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "E-CHALLAN (Vehicle Emission Violation)", ln=True, align="C")
    pdf.ln(10)

    for key, value in challan.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)

    path = f"/tmp/challan_{datetime.now().timestamp()}.pdf"
    pdf.output(path)
    return path

# ================================
# UI
# ================================
st.title("🚗 Vehicle Smoke Detection & Enforcement System")
st.write("AI-powered vehicular pollution monitoring using CCTV simulation.")

camera_id = st.selectbox(
    "📍 Camera Location",
    ["Cam-Delhi-01", "Cam-Delhi-02", "Cam-Delhi-03"]
)

uploaded_file = st.file_uploader(
    "📤 Upload Vehicle Image or CCTV Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# ================================
# IMAGE MODE
# ================================
if uploaded_file and uploaded_file.type.startswith("image"):
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Captured Frame", use_column_width=True)

    smoke_score, status, severity, confidence = detect_smoke(image_bgr)
    vehicle_type = detect_vehicle_type(image_bgr)
    plate, plate_img = detect_number_plate(image_bgr)

    st.subheader("📊 Detection Result")
    st.metric("Smoke Score", f"{smoke_score:.2f}")
    st.metric("Severity", severity)
    st.metric("Confidence", f"{confidence}%")

    st.info(f"Vehicle Type: {vehicle_type}")
    st.info(f"Number Plate: {plate}")

    if plate_img is not None:
        st.image(plate_img, caption="Detected Plate", width=300)

    if status != "Normal Emission":
        st.error("🚨 Pollution Violation Detected")

        challan = {
            "Vehicle Number": plate,
            "Vehicle Type": vehicle_type,
            "Camera": camera_id,
            "Severity": severity,
            "Fine": "₹1000",
            "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        st.session_state.violations.append(challan)

        st.subheader("🧾 Auto Generated e-Challan")
        st.json(challan)

        pdf_path = generate_pdf(challan)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇ Download e-Challan PDF",
                f,
                file_name="e_challan.pdf"
            )
    else:
        st.success("✅ Emission Within Permissible Limit")

# ================================
# VIDEO MODE (First 20 Frames)
# ================================
elif uploaded_file and uploaded_file.type.startswith("video"):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    st.subheader("🎥 CCTV Video Analysis (First 20 Frames)")
    frame_count = 0

    while cap.isOpened() and frame_count < 20:
        ret, frame = cap.read()
        if not ret:
            break

        smoke_score, status, severity, confidence = detect_smoke(frame)

        if status != "Normal Emission":
            plate, _ = detect_number_plate(frame)

            st.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption=f"Frame {frame_count+1} | Smoke Detected",
                use_column_width=True
            )

            st.session_state.violations.append({
                "Vehicle Number": plate,
                "Camera": camera_id,
                "Severity": severity,
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        frame_count += 1

    cap.release()
    st.success("✅ Video processing completed")

# ================================
# DASHBOARD
# ================================
if len(st.session_state.violations) > 0:
    st.subheader("📊 Violation History & Hotspots")

    df = pd.DataFrame(st.session_state.violations)
    st.dataframe(df, use_container_width=True)

    if "Camera" in df.columns:
        hotspot = df["Camera"].value_counts()
        fig, ax = plt.subplots()
        hotspot.plot(kind="bar", ax=ax)
        ax.set_title("Pollution Hotspots (Camera-wise)")
        ax.set_ylabel("Violations")
        st.pyplot(fig)

st.write("🕒 Last Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
