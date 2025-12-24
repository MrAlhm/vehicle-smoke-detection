import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

# -------------------------------
# Smoke Detection Logic
# -------------------------------
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

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Vehicle Smoke Detection System", layout="centered")

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
    st.image(image, caption="Captured Vehicle Frame", use_column_width=True)

    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    smoke_score, smoke_status = detect_smoke(image_bgr)

    st.subheader("📊 Detection Result")
    st.metric("Smoke Score", f"{smoke_score:.2f}")
    st.metric("Smoke Status", smoke_status)

    if smoke_status == "Excessive Smoke":
        st.error("🚨 Polluting Vehicle Detected")
    else:
        st.success("✅ Emission Within Permissible Limit")

    st.write("🕒 Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
