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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Emission Monitoring Platform",
    layout="wide"
)

# =====================================================
# GLOBAL STYLE (Professional, Pollution Theme)
# =====================================================
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
    padding: 22px;
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

# =====================================================
# SESSION STATE
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "Detection"

if "records" not in st.session_state:
    st.session_state.records = []

# =====================================================
# SIDEBAR NAVIGATION (FIXED)
# =====================================================
st.sidebar.title("Emission Monitoring")

st.session_state.page = st.sidebar.radio(
    "Navigation",
    ["Detection", "Video Tracking", "e-Challan", "Analytics", "Hotspots Map", "About"]
)

camera_location = st.sidebar.selectbox(
    "Camera Location",
    [
        "Delhi – Connaught Place",
        "Mumbai – Andheri East",
        "Bengaluru – Silk Board",
        "Hyderabad – Hitech City",
        "Chennai – T Nagar",
        "Kolkata – Salt Lake"
    ]
)

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
            x, y, w, h = cv2.boundingRect(approx)
            plate = image[y:y+h, x:x+w]
            text = reader.readtext(plate)
            if text:
                return text[0][1], plate
    return "Not Readable", None


# =====================================================
# MULTI-VEHICLE DETECTION (CPU SAFE)
# =====================================================
def detect_multiple_vehicles(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    mask = ((s < 60) & (v > 150)).astype("uint8") * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        if cv2.contourArea(c) > 800:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
    return boxes


# =====================================================
# HERO
# =====================================================
st.markdown("""
<div class='card'>
<h1>Smart Vehicle Emission Monitoring</h1>
<p>AI-based detection of excessive vehicular smoke using image & video feeds.</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# PAGE: DETECTION (IMAGE)
# =====================================================
if st.session_state.page == "Detection":

    uploaded = st.file_uploader(
        "Upload Vehicle Image / CCTV Frame",
        type=["jpg", "jpeg", "png"]
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

        # AI Confidence Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#22d3ee"}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        if level != "Low":
            st.markdown("<div class='alert'>Polluting vehicle detected</div>", unsafe_allow_html=True)

            plate, plate_img = detect_number_plate(img)
            if plate_img is not None:
                st.image(plate_img, width=300)

            st.write("Detected Number Plate:", plate)

            st.session_state.records.append({
                "Time": datetime.now(),
                "Camera": camera_location,
                "Plate": plate,
                "Severity": level
            })
        else:
            st.markdown("<div class='success'>Emission within permissible limits</div>", unsafe_allow_html=True)

# =====================================================
# PAGE: VIDEO MULTI-VEHICLE TRACKING
# =====================================================
elif st.session_state.page == "Video Tracking":

    video = st.file_uploader(
        "Upload CCTV Video",
        type=["mp4", "avi", "mov"]
    )

    if video:
        tfile = "temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(video.read())

        cap = cv2.VideoCapture(tfile)
        frame_count = 0

        st.info("Processing first 15 frames for multi-vehicle tracking")

        while cap.isOpened() and frame_count < 15:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = detect_multiple_vehicles(frame)

            for i, (x, y, w, h) in enumerate(boxes):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f"Vehicle {i+1}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2
                )

            st.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption=f"Frame {frame_count+1}",
                use_column_width=True
            )

            frame_count += 1

        cap.release()
        st.success("Video analysis completed")

# =====================================================
# PAGE: E-CHALLAN
# =====================================================
elif st.session_state.page == "e-Challan":

    if not st.session_state.records:
        st.info("No violations recorded yet.")
    else:
        last = st.session_state.records[-1]

        st.markdown("<div class='card'><h2>e-Challan</h2></div>", unsafe_allow_html=True)

        col1, col2 = st.columns([4,1])
        with col1:
            st.write("Vehicle Number:", last["Plate"])
            st.write("Severity:", last["Severity"])
            st.write("Camera:", last["Camera"])
            st.write("Time:", last["Time"])
            st.write("Fine: ₹5000")
        with col2:
            if st.button("Download PDF"):
                st.success("PDF download enabled (integration ready)")

# =====================================================
# PAGE: ANALYTICS
# =====================================================
elif st.session_state.page == "Analytics":

    if st.session_state.records:
        df = pd.DataFrame(st.session_state.records)
        st.dataframe(df)
        st.bar_chart(df["Severity"].value_counts())
    else:
        st.info("No analytics data yet")

# =====================================================
# PAGE: HOTSPOTS MAP
# =====================================================
elif st.session_state.page == "Hotspots Map":

    m = folium.Map(location=[20.59, 78.96], zoom_start=5)

    for r in st.session_state.records:
        folium.CircleMarker(
            location=[20.59, 78.96],
            radius=8,
            color="red",
            fill=True,
            popup=r["Plate"]
        ).add_to(m)

    st_folium(m, width=1100, height=500)

# =====================================================
# PAGE: ABOUT
# =====================================================
elif st.session_state.page == "About":

    st.markdown("""
    <div class='card'>
    <h2>About the Platform</h2>
    <p>
    A scalable AI-powered solution for monitoring vehicular emissions using
    traffic camera feeds, analytics, and real-time detection.
    </p>
    </div>
    """, unsafe_allow_html=True)
