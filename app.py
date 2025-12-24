import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import easyocr
import tempfile
import matplotlib.pyplot as plt

# ================================
# Initialize OCR (DL Model)
# ================================
reader = easyocr.Reader(['en'], gpu=False)

# ================================
# Smoke Detection
# ================================
def detect_smoke(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    smoke_mask = (s < 60) & (v > 150)
    smoke_score = np.sum(smoke_mask) / smoke_mask.size

    if smoke_score >= 0.30:
        severity = "High"
        status = "Excessive Smoke"
    else:
        severity = "Low"
        status = "Normal Emission"

    confidence = int(min(100, smoke_score * 200))

    return smoke_score, status, severity, confidence

# ================================
# Number Plate Detection (Demo)
# ================================
def detect_number_plate(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    plate_img = None

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate_img = image_bgr[y:y+h, x:x+w]
            break

    if plate_img is None:
        return "Not Readable", None

    result = reader.readtext(plate_img)
    if len(result) == 0:
        return "Not Readable", plate_img

    return result[0][1], plate_img

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Vehicle Smoke Detection System", layout="centered")

st.title("🚗 Vehicle Smoke Detection System")
st.write("AI-based real-time smoke monitoring using CCTV simulation.")

camera_id = st.selectbox(
    "📍 Camera Location",
    ["Cam-Delhi-01", "Cam-Delhi-02", "Cam-Delhi-03"]
)

uploaded_file = st.file_uploader(
    "📤 Upload Vehicle Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# ================================
# Session Storage
# ================================
if "violations" not in st.session_state:
    st.session_state.violations = []

# ================================
# IMAGE MODE
# ================================
if uploaded_file and uploaded_file.type.startswith("image"):
    image = Image.open(uploaded_file)
    st.image(image, caption="Captured Frame", use_column_width=True)

    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    smoke_score, status, severity, confidence = detect_smoke(image_bgr)

    st.subheader("📊 Detection Result")
    st.metric("Smoke Score", f"{smoke_score:.2f}")
    st.metric("Severity", severity)
    st.metric("Confidence", f"{confidence}%")

    if status == "Excessive Smoke":
        st.error("🚨 Polluting Vehicle Detected")

        plate, plate_img = detect_number_plate(image_bgr)

        if plate_img is not None:
            st.image(plate_img, caption="Cropped Plate Region", width=300)

        st.info(f"Detected Number Plate: {plate}")

        st.session_state.violations.append({
            "Plate": plate,
            "Camera": camera_id,
            "Severity": severity,
            "Time": datetime.now()
        })
    else:
        st.success("✅ Emission Within Permissible Limit")

# ================================
# VIDEO MODE (FIRST 20 FRAMES)
# ================================
elif uploaded_file and uploaded_file.type.startswith("video"):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    frame_count = 0

    st.subheader("🎥 CCTV Frame Processing (First 20 Frames)")

    while cap.isOpened() and frame_count < 20:
        ret, frame = cap.read()
        if not ret:
            break

        smoke_score, status, severity, confidence = detect_smoke(frame)

        if status == "Excessive Smoke":
            plate, _ = detect_number_plate(frame)

            st.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption=f"Frame {frame_count+1} | Smoke Detected",
                use_column_width=True
            )

            st.session_state.violations.append({
                "Plate": plate,
                "Camera": camera_id,
                "Severity": severity,
                "Time": datetime.now()
            })

        frame_count += 1

    cap.release()
    st.success("✅ Video analysis completed (20-frame limit)")

# ================================
# HOTSPOT DASHBOARD
# ================================
if len(st.session_state.violations) > 0:
    st.subheader("📍 Pollution Hotspot Dashboard")

    df = pd.DataFrame(st.session_state.violations)

    st.dataframe(df)

    hotspot = df["Camera"].value_counts()

    fig, ax = plt.subplots()
    hotspot.plot(kind="bar", ax=ax)
    ax.set_title("Violations per Camera Location")
    ax.set_ylabel("Number of Violations")

    st.pyplot(fig)

# ================================
# FOOTER
# ================================
st.write("🕒 Last Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
