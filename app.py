import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import easyocr
from fpdf import FPDF

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Emission Detection Platform",
    layout="wide"
)

# =====================================================
# GLOBAL CSS – PROFESSIONAL STARTUP THEME
# =====================================================
st.markdown("""
<style>

/* Base */
html, body {
    background-color: #0e1117;
    font-family: 'Inter', sans-serif;
}

/* Remove Streamlit branding */
#MainMenu, footer, header {
    visibility: hidden;
}

/* Hero section */
.hero {
    background: linear-gradient(135deg, #0b3c49, #1b6b63);
    padding: 60px;
    border-radius: 25px;
    color: white;
    margin-bottom: 40px;
}

/* Section card */
.card {
    background-color: #161b22;
    border-radius: 18px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    margin-bottom: 30px;
}

/* Upload box */
.upload-box {
    border: 2px dashed #2dd4bf;
    border-radius: 15px;
    padding: 40px;
    text-align: center;
    color: #9ca3af;
    margin-top: 20px;
}

/* Metric style */
.metric {
    background-color: #0f172a;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}

/* Alert */
.alert {
    background: linear-gradient(90deg, #7f1d1d, #991b1b);
    padding: 18px;
    border-radius: 12px;
    color: white;
    font-size: 18px;
}

/* Success */
.success {
    background: linear-gradient(90deg, #064e3b, #065f46);
    padding: 18px;
    border-radius: 12px;
    color: white;
    font-size: 18px;
}

/* Button */
.download-btn {
    background-color: #2563eb;
    padding: 10px 22px;
    border-radius: 10px;
    color: white;
    font-weight: 600;
}

/* Top navigation */
.nav {
    display: flex;
    gap: 30px;
    margin-bottom: 25px;
    font-weight: 500;
    color: #9ca3af;
}

.nav-item {
    cursor: pointer;
}

.nav-item-active {
    color: #2dd4bf;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================
if "media" not in st.session_state:
    st.session_state.media = None

if "records" not in st.session_state:
    st.session_state.records = []

if "page" not in st.session_state:
    st.session_state.page = "Detection"

# =====================================================
# TOP NAVIGATION (CLEAN, SMALL)
# =====================================================
cols = st.columns([1,1,1,1,1])
pages = ["Detection", "CCTV", "e-Challan", "Analytics", "About"]

for i, p in enumerate(pages):
    if cols[i].button(p, key=p):
        st.session_state.page = p

# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<div class="hero">
<h1>Smart Vehicle Emission Monitoring</h1>
<p>AI-powered detection of excessive vehicular smoke using traffic camera imagery.</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# MODELS
# =====================================================
reader = easyocr.Reader(['en'], gpu=False)

# =====================================================
# CORE FUNCTIONS
# =====================================================
def detect_smoke(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    smoke_mask = (s < 60) & (v > 150)
    score = np.sum(smoke_mask) / smoke_mask.size

    if score >= 0.35:
        level = "High"
    elif score >= 0.20:
        level = "Moderate"
    else:
        level = "Low"

    confidence = min(95, int(score * 200))
    return score, level, confidence


def detect_number_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours[:15]:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate = image[y:y+h, x:x+w]
            result = reader.readtext(plate)
            if result:
                return result[0][1], plate
    return "Not Readable", None

# =====================================================
# PAGE: DETECTION
# =====================================================
if st.session_state.page == "Detection":

    st.markdown("<div class='card'><h2>Upload Vehicle Image or CCTV Frame</h2></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        st.session_state.media = uploaded

    if st.session_state.media:
        image = Image.open(st.session_state.media)
        st.image(image, use_column_width=True)

        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        score, level, confidence = detect_smoke(img_bgr)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric'><h3>Smoke Score</h3><h1>{score:.2f}</h1></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric'><h3>Severity</h3><h1>{level}</h1></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric'><h3>Confidence</h3><h1>{confidence}%</h1></div>", unsafe_allow_html=True)

        if level != "Low":
            st.markdown("<div class='alert'>Polluting vehicle detected</div>", unsafe_allow_html=True)

            plate, plate_img = detect_number_plate(img_bgr)
            if plate_img is not None:
                st.image(plate_img, width=300)

            st.write("Detected Number Plate:", plate)

            st.session_state.records.append({
                "Time": datetime.now(),
                "Plate": plate,
                "Severity": level
            })
        else:
            st.markdown("<div class='success'>Emission within permissible limits</div>", unsafe_allow_html=True)

# =====================================================
# PAGE: E-CHALLAN
# =====================================================
elif st.session_state.page == "e-Challan":

    st.markdown("<div class='card'><h2>e-Challan</h2></div>", unsafe_allow_html=True)

    if not st.session_state.records:
        st.info("No violations recorded yet.")
    else:
        last = st.session_state.records[-1]

        col1, col2 = st.columns([4,1])
        with col1:
            st.write("Vehicle:", last["Plate"])
            st.write("Severity:", last["Severity"])
            st.write("Time:", last["Time"])
            st.write("Fine: ₹5000")
        with col2:
            if st.button("Download PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for k,v in last.items():
                    pdf.cell(0, 10, f"{k}: {v}", ln=True)
                pdf.output("challan.pdf")
                st.success("Downloaded")

# =====================================================
# PAGE: ANALYTICS
# =====================================================
elif st.session_state.page == "Analytics":
    st.markdown("<div class='card'><h2>Analytics</h2></div>", unsafe_allow_html=True)
    if st.session_state.records:
        df = pd.DataFrame(st.session_state.records)
        st.dataframe(df)
        st.bar_chart(df["Severity"].value_counts())
    else:
        st.info("No data yet")

# =====================================================
# PAGE: ABOUT
# =====================================================
elif st.session_state.page == "About":
    st.markdown("""
    <div class='card'>
    <h2>About This Platform</h2>
    <p>
    A professional-grade AI system designed to assist cities in monitoring
    vehicular emissions using computer vision and deep learning.
    </p>
    </div>
    """, unsafe_allow_html=True)
