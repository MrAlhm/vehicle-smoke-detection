import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import easyocr
from ultralytics import YOLO
import pandas as pd

# -------------------------------------------------
# Initialize Models
# -------------------------------------------------
reader = easyocr.Reader(['en'], gpu=False)
yolo_model = YOLO("yolov8n.pt")   # YOLOv8 Nano (CPU friendly)

# -------------------------------------------------
# Session State for Violation History
# -------------------------------------------------
if "violations" not in st.session_state:
    st.session_state.violations = []

# -------------------------------------------------
# Smoke Detection Logic
# -------------------------------------------------
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

    confidence = int(min(100, smoke_score * 200))
    return smoke_score, status, severity, confidence

# -------------------------------------------------
# YOLO Number Plate Detection + Bounding Box + OCR
# -------------------------------------------------
def detect_plate_with_bbox(image_bgr):
    results = yolo_model(image_bgr, conf=0.4)

    plate_img = None
    bbox = None

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = image_bgr[y1:y2, x1:x2]
                bbox = (x1, y1, x2, y2)
                break

    if plate_img is None or plate_img.size == 0:
        return "Not Readable", image_bgr, None, None

    ocr_result = reader.readtext(plate_img)
    plate_text = ocr_result[0][1] if len(ocr_result) > 0 else "Not Readable"

    image_with_box = image_bgr.copy()
    if bbox:
        cv2.rectangle(
            image_with_box,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),
            2
        )

    return plate_text, image_with_box, plate_img, bbox

# -------------------------------------------------
# Vehicle Type Detection (Rule-based for Demo)
# -------------------------------------------------
def detect_vehicle_type(bbox):
    if bbox is None:
        return "Unknown"

    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)

    if area < 15000:
        return "Two-Wheeler"
    elif area < 40000:
        return "Car"
    elif area < 70000:
        return "Bus"
    else:
        return "Truck"

# -------------------------------------------------
# Auto e-Challan Generator
# -------------------------------------------------
def generate_e_challan(plate_number, severity, vehicle_type):
    fine_amount = {
        "Medium": "₹500",
        "High": "₹1000",
        "Low": "₹0"
    }.get(severity, "₹1000")

    challan = {
        "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Vehicle Number": plate_number,
        "Vehicle Type": vehicle_type,
        "Violation": "Excessive Smoke Emission",
        "Severity": severity,
        "Fine Amount": fine_amount
    }
    return challan

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(
    page_title="Vehicle Smoke Detection & Enforcement System",
    layout="wide"
)

st.title("🚗 Vehicle Smoke Detection & Enforcement System")
st.write(
    "AI-powered system for detecting vehicular smoke, identifying vehicles, and generating automated e-challans."
)

uploaded_file = st.file_uploader(
    "📤 Upload Vehicle Image (CCTV Frame Simulation)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    smoke_score, smoke_status, severity, confidence = detect_smoke(image_bgr)

    st.subheader("📊 Smoke Analysis")
    st.metric("Smoke Score", f"{smoke_score:.2f}")
    st.metric("Smoke Severity", severity)
    st.metric("Detection Confidence", f"{confidence}%")

    if smoke_status != "Normal Emission":
        st.error("🚨 Polluting Vehicle Detected")

        plate_text, boxed_image, plate_crop, bbox = detect_plate_with_bbox(image_bgr)
        vehicle_type = detect_vehicle_type(bbox)

        with col2:
            st.image(
                cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB),
                caption="YOLO Bounding Box (Number Plate)",
                use_column_width=True
            )

        if plate_crop is not None:
            st.image(
                cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB),
                caption="Cropped Plate Region",
                width=300
            )

        st.info(f"Detected Number Plate: {plate_text}")
        st.info(f"Vehicle Type: {vehicle_type}")

        challan = generate_e_challan(plate_text, severity, vehicle_type)
        st.subheader("🧾 Auto-Generated e-Challan")

        for key, value in challan.items():
            st.write(f"**{key}:** {value}")

        st.session_state.violations.append(challan)

    else:
        st.success("✅ Emission Within Permissible Limit")

# -------------------------------------------------
# Violation History Dashboard
# -------------------------------------------------
st.subheader("📊 Violation History Dashboard")

if len(st.session_state.violations) > 0:
    df = pd.DataFrame(st.session_state.violations)
    st.dataframe(df, use_container_width=True)
    st.metric("Total Violations Recorded", len(df))
else:
    st.info("No violations recorded yet.")
