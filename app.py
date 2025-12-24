import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import easyocr
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Smart Emission Monitoring Platform",
    layout="wide"
)

# ======================================================
# GLOBAL STYLE (Enterprise Grade)
# ======================================================
st.markdown("""
<style>
html, body { background-color: #0b1220; }
#MainMenu, footer, header { visibility: hidden; }

.sidebar .sidebar-content {
    background-color: #0f172a;
}

h1, h2, h3 { color: #e5e7eb; }
p { color: #9ca3af; }

.card {
    background: #111827;
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.metric {
    background: #020617;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
}

.alert {
    background: linear-gradient(90deg, #7f1d1d, #991b1b);
    padding: 16px;
    border-radius: 12px;
    color: white;
}

.success {
    background: linear-gradient(90deg, #064e3b, #065f46);
    padding: 16px;
    border-radius: 12px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SESSION STATE
# ======================================================
if "page" not in st.session_state:
    st.session_state.page = "Detection"

if "records" not in st.session_state:
    st.session_state.records = []

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
st.sidebar.title("Smart City Dashboard")

st.session_state.page = st.sidebar.radio(
    "Navigate",
    ["Detection", "Live CCTV", "e-Challan", "Analytics", "Hotspots Map", "About"]
)

camera_location = st.sidebar.selectbox(
    "Camera Location",
    ["Delhi-01", "Mumbai-02", "Bengaluru-03", "Hyderabad-04"]
)

# ======================================================
# MODELS
# ======================================================
reader = easyocr.Reader(['en'], gpu=False)

# ======================================================
# FUNCTIONS
# ======================================================
def detect_smoke(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = (s < 60) & (v > 150)
    score = np.sum(mask) / mask.size

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

    for cnt in contours[:20]:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            plate = image[y:y+h, x:x+w]
            result = reader.readtext(plate)
            if result:
                return result[0][1], plate
    return "Not Readable", None

# ======================================================
# PAGE: DETECTION
# ======================================================
if st.session_state.page == "Detection":

    st.markdown("<div class='card'><h1>Vehicle Emission Detection</h1></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload Vehicle Image / CCTV Frame",
        type=["jpg","jpeg","png"]
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_column_width=True)

        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        score, level, confidence = detect_smoke(img)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric'><h3>Smoke Score</h3><h1>{score:.2f}</h1></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric'><h3>Severity</h3><h1>{level}</h1></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric'><h3>Confidence</h3><h1>{confidence}%</h1></div>", unsafe_allow_html=True)

        # AI Confidence Graph
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#22d3ee"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        if level != "Low":
            st.markdown("<div class='alert'>Polluting Vehicle Detected</div>", unsafe_allow_html=True)
            plate, plate_img = detect_number_plate(img)

            if plate_img is not None:
                st.image(plate_img, width=300)

            st.write("Number Plate:", plate)

            st.session_state.records.append({
                "Time": datetime.now(),
                "Camera": camera_location,
                "Plate": plate,
                "Severity": level
            })
        else:
            st.markdown("<div class='success'>Emission within permissible limits</div>", unsafe_allow_html=True)

# ======================================================
# PAGE: HOTSPOTS MAP (REAL MAP)
# ======================================================
elif st.session_state.page == "Hotspots Map":

    st.markdown("<div class='card'><h1>Pollution Hotspots</h1></div>", unsafe_allow_html=True)

    m = folium.Map(location=[28.61, 77.20], zoom_start=5)

    for rec in st.session_state.records:
        folium.CircleMarker(
            location=[28.61, 77.20],
            radius=8,
            color="red",
            fill=True,
            popup=rec["Plate"]
        ).add_to(m)

    st_folium(m, width=1000, height=500)

# ======================================================
# PAGE: ANALYTICS
# ======================================================
elif st.session_state.page == "Analytics":

    st.markdown("<div class='card'><h1>Violation Analytics</h1></div>", unsafe_allow_html=True)

    if st.session_state.records:
        df = pd.DataFrame(st.session_state.records)
        st.dataframe(df)
        st.bar_chart(df["Severity"].value_counts())
    else:
        st.info("No data yet")

# ======================================================
# PAGE: ABOUT
# ======================================================
elif st.session_state.page == "About":

    st.markdown("""
    <div class='card'>
    <h1>About This Platform</h1>
    <p>
    A scalable AI-powered smart city solution for real-time vehicular emission
    monitoring, analytics, and enforcement.
    </p>
    </div>
    """, unsafe_allow_html=True)
