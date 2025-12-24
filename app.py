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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Vehicle Smoke Detection",
    layout="wide",
    page_icon="🚦"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
}
.sub-title {
    font-size: 18px;
    color: #555;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    text-align: center;
}
.badge-red {
    color: white;
    background-color: #e74c3c;
    padding: 8px 16px;
    border-radius: 20px;
}
.badge-green {
    color: white;
    background-color: #2ecc71;
    padding: 8px 16px;
    border-radius: 20px;
}
.badge-yellow {
    color: black;
    background-color: #f1c40f;
    padding: 8px 16px;
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🚗 Smart Vehicle Smoke Detection & Enforcement</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered pollution monitoring for Smart Cities</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- OCR ----------------
reader = easyocr.Reader(['en'], gpu=False)

# ---------------- SESSION ----------------
if "violations" not in st.session_state:
    st.session_state.violations = []

# ---------------- FUNCTIONS ----------------
def detect_smoke(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    smoke_mask = (s < 60) & (v > 150)
    score = np.sum(smoke_mask) / smoke_mask.size

    if score >= 0.30:
        return score, "High", "🚨 Violation"
    elif score >= 0.20:
        return score, "Medium", "⚠️ Warning"
    else:
        return score, "Low", "✅ Clean"

def detect_plate(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            plate = image_bgr[y:y+h, x:x+w]
            text = reader.readtext(plate)
            return text[0][1] if text else "Not Readable", plate
    return "Not Readable", None

# ---------------- TABS ----------------
tabs = st.tabs(["🔍 Detection", "🧾 e-Challan", "📊 Dashboard", "ℹ️ About"])

# ========== TAB 1: DETECTION ==========
with tabs[0]:
    cam = st.selectbox("📍 Camera Location", ["Cam-Delhi-01", "Cam-Delhi-02", "Cam-Delhi-03"])
    file = st.file_uploader("Upload Vehicle Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file)
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        score, severity, status = detect_smoke(img_bgr)
        plate, plate_img = detect_plate(img_bgr)

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='card'><h3>Smoke Score</h3><h2>{score:.2f}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='card'><h3>Severity</h3><h2>{severity}</h2></div>", unsafe_allow_html=True)

        if status.startswith("🚨"):
            col3.markdown("<div class='card badge-red'>VIOLATION</div>", unsafe_allow_html=True)
        elif status.startswith("⚠️"):
            col3.markdown("<div class='card badge-yellow'>WARNING</div>", unsafe_allow_html=True)
        else:
            col3.markdown("<div class='card badge-green'>CLEAN</div>", unsafe_allow_html=True)

        st.image(img, caption="Captured Frame", use_column_width=True)

        if status.startswith("🚨"):
            st.error("Polluting Vehicle Detected")
            st.info(f"Number Plate: {plate}")
            st.session_state.violations.append({
                "Plate": plate,
                "Camera": cam,
                "Severity": severity,
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

# ========== TAB 2: E-CHALLAN ==========
with tabs[1]:
    st.subheader("🧾 Generated e-Challans")
    if st.session_state.violations:
        st.table(pd.DataFrame(st.session_state.violations))
    else:
        st.info("No violations yet")

# ========== TAB 3: DASHBOARD ==========
with tabs[2]:
    if st.session_state.violations:
        df = pd.DataFrame(st.session_state.violations)
        st.bar_chart(df["Camera"].value_counts())
    else:
        st.info("No data available")

# ========== TAB 4: ABOUT ==========
with tabs[3]:
    st.markdown("""
### 🚦 Smart Vehicle Smoke Detection System
**Technologies Used**
- Computer Vision
- EasyOCR
- OpenCV
- Streamlit
- AI-based Heuristics

**Purpose**
Reduce air pollution by automated real-time vehicular emission enforcement.
""")

st.markdown("---")
st.caption("© Smart City AI | Hackathon Prototype")
