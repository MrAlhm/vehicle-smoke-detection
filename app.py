import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import easyocr
from ultralytics import YOLO
import pandas as pd

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="Vehicle Smoke Detection System",
    layout="wide"
)

# --------------------------------
# Load Models (Cached)
# --------------------------------
@st.cache_resource
def load_models():
    vehicle_model = YOLO("yolov8n.pt")          # vehicle detection
    plate_model = YOLO("yolov8n.pt")            # reuse for demo (plate via heuristics)
    reader = easyocr.Reader(['en'], gpu=False)
    return vehicle_model, plate_model, reader

vehicle_model, plate_model, reader = load_models()

# --------------------------------
# Smoke Detection
# --------------------------------
def detect_smoke(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    smoke_mask = (s < 60) & (v > 150)
    smoke_score = np.sum(smoke_mask) / smoke_mask.size

    if smoke_score >= 0.30:
        severity = "High"
        status = "Excessive Smoke"
    else:
        severity = "Low"
        status = "Normal Emission"

    confidence = min(int(smoke_score * 200), 100)

    return smoke_score, severity, confidence, status

# --------------------------------
# Vehicle Type Detection (YOLO)
# --------------------------------
def detect_vehicle_type(image_bgr):
    results = vehicle_model(image_bgr, conf=0.4)

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = vehicle_model.names[cls_id]

                if cls_name in ["car", "truck", "bus", "motorcycle"]:
                    return cls_name.capitalize(), box.xyxy[0]

    return "Unknown", None

# --------------------------------
# Number Plate Detection + OCR
# --------------------------------
def detect_number_plate(image_bgr, bbox):
    if bbox is None:
        return "Not Readable", None

    x1, y1, x2, y2 = map(int, bbox)
    plate_img = image_bgr[y1:y2, x1:x2]

    if plate_img.size == 0:
        return "Not Readable", None

    result = reader.readtext(plate_img)

    if len(result) == 0:
        return "Not Readable", plate_img

    return result[0][1], plate_img

# --------------------------------
# Violation History (Session)
# --------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------
# UI
# --------------------------------
st.title("🚗 Vehicle Smoke Detection System")
st.write("AI-powered real-time vehicular pollution monitoring using computer vision")

uploaded_file = st.file_uploader(
    "📤 Upload Vehicle Image (CCTV Frame Simulation)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Captured Vehicle Frame", use_column_width=True)

    # ---- Detection Pipeline ----
    smoke_score, severity, confidence, smoke_status = detect_smoke(image_bgr)
    vehicle_type, bbox = detect_vehicle_type(image_bgr)
    plate_number, plate_crop = detect_number_plate(image_bgr, bbox)

    # ---- Draw Bounding Box ----
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    st.subheader("📊 Detection Result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Smoke Score", f"{smoke_score:.2f}")
    col2.metric("Smoke Severity", severity)
    col3.metric("Detection Confidence", f"{confidence}%")

    if smoke_status == "Excessive Smoke":
        st.error("🚨 Polluting Vehicle Detected")
    else:
        st.success("✅ Emission Within Permissible Limit")

    st.subheader("🚘 Vehicle Identification")
    st.info(f"Detected Number Plate: {plate_number}")
    st.info(f"Vehicle Type: {vehicle_type}")

    if plate_crop is not None:
        st.image(plate_crop, caption="Cropped Plate Region", width=300)

    # ---- e-Challan ----
    if smoke_status == "Excessive Smoke":
        st.subheader("🧾 Auto-Generated e-Challan")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        challan = {
            "Number Plate": plate_number,
            "Vehicle Type": vehicle_type,
            "Smoke Score": round(smoke_score, 2),
            "Severity": severity,
            "Date & Time": timestamp
        }

        st.json(challan)

        st.session_state.history.append(challan)

    st.write("🕒 Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# --------------------------------
# Dashboard
# --------------------------------
if len(st.session_state.history) > 0:
    st.subheader("📈 Violation History Dashboard")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)
