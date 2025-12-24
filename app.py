import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import easyocr

# --------------------------------
# Initialize OCR Reader (DL Model)
# --------------------------------
reader = easyocr.Reader(['en'], gpu=False)

# --------------------------------
# Smoke Detection Function
# --------------------------------
def detect_smoke(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    smoke_mask = (s < 60) & (v > 150)
    smoke_score = np.sum(smoke_mask) / smoke_mask.size

    if smoke_score >= 0.30:
        status = "Excessive Smoke"
    else:
        status = "Normal Emission"

    return smoke_score, status

# --------------------------------
# Smoke Severity + Confidence
# --------------------------------
def get_smoke_severity_and_confidence(smoke_score):
    if smoke_score < 0.15:
        severity = "Normal"
    elif smoke_score < 0.30:
        severity = "Mild"
    elif smoke_score < 0.50:
        severity = "High"
    else:
        severity = "Severe"

    confidence = min(100, int(smoke_score * 200))
    return severity, confidence

# --------------------------------
# Number Plate Detection Function
# --------------------------------
def detect_number_plate(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    plate_img = None

    for cnt in contours:
        approx = cv2.approxPolyDP(
            cnt, 0.018 * cv2.arcLength(cnt, True), True
        )
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate_img = image_bgr[y:y+h, x:x+w]
            break

    if plate_img is None:
        return "Not Readable"

    result = reader.readtext(plate_img)

    if len(result) == 0:
        return "Not Readable"

    return result[0][1]

# --------------------------------
# Streamlit UI
# --------------------------------
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

# ✅ Demo Mode Toggle
demo_mode = st.checkbox("🎥 Demo Mode: Force Number Plate Detection")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Captured Vehicle Frame", use_column_width=True)

    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    smoke_score, smoke_status = detect_smoke(image_bgr)
    severity, confidence = get_smoke_severity_and_confidence(smoke_score)

    st.subheader("📊 Detection Result")
    st.metric("Smoke Score", f"{smoke_score:.2f}")
    st.metric("Smoke Severity", severity)
    st.metric("Detection Confidence", f"{confidence}%")

    # 🚨 Enforcement Logic
    if smoke_status == "Excessive Smoke":
        st.error("🚨 Polluting Vehicle Detected")

    # 🚘 Number Plate Detection (Normal or Demo Mode)
    if smoke_status == "Excessive Smoke" or demo_mode:
        plate_number = detect_number_plate(image_bgr)
        st.subheader("🚘 Vehicle Identification")
        st.info(f"Detected Number Plate: {plate_number}")
    else:
        st.success("✅ Emission Within Permissible Limit")

    st.write(
        "🕒 Timestamp:",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
