import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import easyocr

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Traffic Police Smoke Monitoring System",
    layout="wide",
    page_icon="🚓"
)

# =====================================================
# CUSTOM POLICE THEME CSS
# =====================================================
st.markdown("""
<style>
body {
    background-color: #0b132b;
}
.main {
    background-color: #0b132b;
}
.sidebar .sidebar-content {
    background-color: #1c2541;
}
h1, h2, h3, h4 {
    color: #ffffff;
}
.metric-label {
    color: #ffffff !important;
}
.metric-value {
    color: #fca311 !important;
}
.alert-box {
    background: linear-gradient(90deg, #9b2226, #ae2012);
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 18px;
}
.success-box {
    background: linear-gradient(90deg, #2a9d8f, #40916c);
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 18px;
}
.card {
    background-color: #1c2541;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    box-shadow: 0 0 10px rgba(0,0,0,0.4);
}
.footer {
    text-align:center;
    color:#adb5bd;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================
if "image" not in st.session_state:
    st.session_state.image = None

if "records" not in st.session_state:
    st.session_state.records = []

# =====================================================
# SIDEBAR (POLICE CONSOLE)
# =====================================================
st.sidebar.markdown("## 🚓 Traffic Police Console")

page = st.sidebar.radio(
    "Navigation",
    ["🚨 Detection", "📹 CCTV", "🧾 e-Challan", "📊 Dashboard", "🔥 Hotspots", "ℹ About"]
)

camera_location = st.sidebar.selectbox(
    "📍 Camera Location",
    [
        "Delhi – Connaught Place",
        "Delhi – Anand Vihar",
        "Mumbai – Andheri East",
        "Mumbai – Bandra West",
        "Bengaluru – Silk Board",
        "Chennai – T Nagar",
        "Hyderabad – Hitech City",
        "Kolkata – Salt Lake"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload CCTV Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.session_state.image = uploaded_file

# =====================================================
# MODELS
# =====================================================
reader = easyocr.Reader(['en'], gpu=False)

# =====================================================
# FUNCTIONS
# =====================================================
def detect_smoke(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    smoke_mask = (s < 60) & (v > 150)
    score = np.sum(smoke_mask) / smoke_mask.size

    if score >= 0.35:
        severity = "HIGH"
        status = "POLLUTING"
    elif score >= 0.20:
        severity = "MEDIUM"
        status = "SUSPICIOUS"
    else:
        severity = "LOW"
        status = "NORMAL"

    confidence = min(95, int(score * 200))
    return score, severity, status, confidence


def detect_number_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours[:15]:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate = image[y:y+h, x:x+w]
            text = reader.readtext(plate)
            if text:
                return text[0][1], plate
    return "Not Readable", None


def detect_vehicle_type(image):
    h, w, _ = image.shape
    if w > 1100:
        return "Truck / Bus"
    elif w > 700:
        return "Car"
    else:
        return "Two-Wheeler"

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="card">
<h1>🚓 Traffic Police Vehicle Smoke Monitoring System</h1>
<p>AI-Powered Enforcement | Smart City Pollution Control</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# PAGE: DETECTION
# =====================================================
if page == "🚨 Detection":

    if st.session_state.image is None:
        st.warning("📤 Upload a CCTV image from sidebar")
    else:
        img = Image.open(st.session_state.image)
        st.image(img, caption="📸 CCTV Captured Vehicle", use_column_width=True)

        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        score, severity, status, confidence = detect_smoke(img_bgr)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Smoke Score", f"{score:.2f}")
        col2.metric("Severity", severity)
        col3.metric("Status", status)
        col4.metric("Confidence", f"{confidence}%")

        if status != "NORMAL":
            st.markdown('<div class="alert-box">🚨 POLLUTING VEHICLE DETECTED</div>', unsafe_allow_html=True)

            plate, plate_img = detect_number_plate(img_bgr)
            vehicle = detect_vehicle_type(img_bgr)

            if plate_img is not None:
                st.image(plate_img, caption="🔍 Cropped Number Plate", width=300)

            st.info(f"🚘 Vehicle Number: {plate}")
            st.info(f"🚚 Vehicle Type: {vehicle}")
            st.info(f"📍 Camera: {camera_location}")

            st.session_state.records.append({
                "Time": datetime.now(),
                "Camera": camera_location,
                "Plate": plate,
                "Vehicle": vehicle,
                "Severity": severity
            })
        else:
            st.markdown('<div class="success-box">✅ EMISSION WITHIN LEGAL LIMIT</div>', unsafe_allow_html=True)

# =====================================================
# PAGE: E-CHALLAN
# =====================================================
elif page == "🧾 e-Challan":
    st.subheader("🧾 Auto-Generated Traffic Police e-Challan")

    if not st.session_state.records:
        st.warning("No violations recorded")
    else:
        last = st.session_state.records[-1]
        st.json({
            "Date & Time": last["Time"].strftime("%Y-%m-%d %H:%M:%S"),
            "Camera": last["Camera"],
            "Vehicle No": last["Plate"],
            "Vehicle Type": last["Vehicle"],
            "Offence": "Excessive Smoke Emission",
            "Fine": "₹5000"
        })

# =====================================================
# PAGE: DASHBOARD
# =====================================================
elif page == "📊 Dashboard":
    st.subheader("📊 Police Analytics Dashboard")

    if st.session_state.records:
        df = pd.DataFrame(st.session_state.records)
        st.dataframe(df)
        st.bar_chart(df["Severity"].value_counts())
    else:
        st.warning("No data yet")

# =====================================================
# PAGE: HOTSPOTS
# =====================================================
elif page == "🔥 Hotspots":
    st.subheader("🔥 Pollution Hotspot Analysis")

    if st.session_state.records:
        df = pd.DataFrame(st.session_state.records)
        st.bar_chart(df["Camera"].value_counts())
    else:
        st.warning("No hotspot data")

# =====================================================
# PAGE: ABOUT
# =====================================================
elif page == "ℹ About":
    st.markdown("""
    ### 🚓 About This System
    - Designed for Indian Traffic Police
    - AI-based real-time pollution enforcement
    - Number plate recognition
    - Smart city analytics
    - Hackathon-ready prototype
    """)

# =====================================================
# FOOTER
# =====================================================
st.markdown("<div class='footer'>© Traffic Police AI | Smart City Initiative</div>", unsafe_allow_html=True)
