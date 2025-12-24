import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import easyocr
from ultralytics import YOLO

# -------------------------------------------------
# Initialize Models
# -------------------------------------------------
reader = easyocr.Reader(['en'], gpu=False)
yolo_model = YOLO("yolov8n.pt")   # YOLOv8 Nano (CPU friendly)

# -------------------------------------------------
# Smoke Detection Logic
# -------------------------------------------------
def detect_smoke(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

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
        return "Not Readable", image_bgr, None

    # OCR
    ocr_result = reader.readtext(plate_img)
    plate_text = ocr_result[0][1] if len(ocr_result) > 0 else "Not Readable"

    # Draw bounding box
    image_with_box = image_bgr.copy()
    if bbox:
        cv2.rectangle(
            image_with_box,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),
            2
        )
        cv2.putText(
            image_with_box,
            "Number Plate",
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return plate_text, image_with_box, plate_img

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(
    page_title="Vehicle Smoke Detection System",
    layout="centered"
)

st.title("🚗 Vehicle Smoke Detection System")
st.write(
    "AI-based system to detect excessive vehicular smoke using traffic camera images."
)

uploaded_file = st.file_uploader(
    "📤 Upload Vehicle Image (CCTV Frame Simulation)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Original Image", use_column_width=True)

    smoke_score, smoke_status, severity, confidence = detect_smoke(image_bgr)

    st.subheader("📊 Smoke Analysis")
    st.metric("Smoke Score", f"{smoke_score:.2f}")
    st.metric("Smoke Severity", severity)
    st.metric("Detection Confidence", f"{confidence}%")

    if smoke_status != "Normal Emission":
        st.error("🚨 Polluting Vehicle Detected")

        plate_text, boxed_image, plate_crop = detect_plate_with_bbox(image_bgr)

        st.subheader("🚘 Number Plate Detection")

        st.image(
            cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB),
            caption="Detected Number Plate (Bounding Box)",
            use_column_width=True
        )

        if plate_crop is not None:
            st.image(
                cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB),
                caption="Cropped Plate Region",
                width=300
            )

        st.info(f"Detected Number Plate: {plate_text}")
    else:
        st.success("✅ Emission Within Permissible Limit")

    st.write(
        "🕒 Timestamp:",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
