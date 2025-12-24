import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import easyocr
from fpdf import FPDF
import plotly.graph_objects as go
import uuid
import tempfile

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Vehicle Smoke Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# BACKGROUND – RACE CAR POLLUTION THEME
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-image:
        linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
        url("https://images.unsplash.com/photo-1519681393784-d120267933ba");
    background-size: cover;
    background-attachment: fixed;
}
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("System Modules")

module = st.sidebar.radio(
    "Select",
    ["Detection", "e-Challan", "Dashboard", "About"]
)

camera_location = st.sidebar.selectbox(
    "Camera Location",
    ["Delhi-01", "Mumbai-02", "Bengaluru-03", "Chennai-04"]
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div style="padding:30px;border-radius:18px;
background:linear-gradient(135deg,#1f4037,#99f2c8)">
<h1 style="color:black;">Smart Vehicle Smoke Monitoring</h1>
<p style="color:black;font-size:17px;">
AI-powered pollution detection using traffic cameras
</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# --------------------------------------------------
# AI MODELS
# --------------------------------------------------
reader = easyocr.Reader(['en'], gpu=False)

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------
def detect_smoke(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    mask = (s < 60) & (v > 150)
    score = np.sum(mask) / mask.size

    if score > 0.30:
        level = "High"
    elif score > 0.18:
        level = "Medium"
    else:
        level = "Low"

    confidence = int(min(95, score * 200))
    return score, level, confidence, mask

def detect_plate(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)

    for r in results:
        if len(r[1]) >= 6:
            return r[1].upper()
    return "Not Readable"

def estimate_vehicle_type(img):
    h, w, _ = img.shape
    if w > 1000:
        return "Bus / Truck"
    elif w > 650:
        return "Car"
    else:
        return "Two-Wheeler"

def confidence_gauge(val):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={"text": "AI Confidence"},
        gauge={"axis": {"range": [0, 100]}}
    ))
    fig.update_layout(height=260)
    return fig

def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0,10,"AUTO-GENERATED E-CHALLAN",ln=True,align="C")
    pdf.ln(8)
    for k,v in data.items():
        pdf.cell(0,8,f"{k}: {v}",ln=True)
    fname = f"e_challan_{data['Violation ID']}.pdf"
    pdf.output(fname)
    return fname

# --------------------------------------------------
# DETECTION MODULE
# --------------------------------------------------
if module == "Detection":

    file = st.file_uploader(
        "Upload Image or Video",
        type=["jpg","jpeg","png","mp4"]
    )

    if file:
        if file.type.startswith("image"):
            image = Image.open(file)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frames = [frame]

        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            cap = cv2.VideoCapture(tfile.name)
            frames = []
            count = 0
            while cap.isOpened() and count < 5:
                ret, f = cap.read()
                if not ret:
                    break
                frames.append(f)
                count += 1
            cap.release()

        for frame in frames[:1]:
            st.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),use_column_width=True)

            score, level, conf, mask = detect_smoke(frame)
            plate = detect_plate(frame)
            vtype = estimate_vehicle_type(frame)

            c1,c2,c3 = st.columns(3)
            c1.metric("Smoke Score", f"{score:.2f}")
            c2.metric("Severity", level)
            c3.metric("Vehicle Type", vtype)

            st.plotly_chart(confidence_gauge(conf), use_container_width=True)

            if level == "High":
                st.error("Polluting Vehicle Detected")

                st.info(f"Detected Plate: {plate}")

                st.session_state["violation"] = {
                    "Violation ID": str(uuid.uuid4())[:8],
                    "Plate": plate,
                    "Vehicle Type": vtype,
                    "Severity": level,
                    "Camera": camera_location,
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                st.success("Emission Within Permissible Limit")

# --------------------------------------------------
# E-CHALLAN MODULE
# --------------------------------------------------
elif module == "e-Challan":
    if "violation" in st.session_state:
        data = st.session_state["violation"]
        for k,v in data.items():
            st.write(f"**{k}:** {v}")

        if st.button("Download e-Challan"):
            file = generate_pdf(data)
            with open(file,"rb") as f:
                st.download_button(
                    "Download PDF",
                    f,
                    file_name=file,
                    mime="application/pdf"
                )
    else:
        st.warning("No violation detected yet.")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
elif module == "Dashboard":
    st.metric("Vehicles Analysed", 1840)
    st.metric("Violations", 412)
    st.metric("High-Risk Cameras", 6)

# --------------------------------------------------
# ABOUT
# --------------------------------------------------
else:
    st.write("""
This system demonstrates an AI-based vehicle smoke monitoring solution.

Prototype:
• Image / video analysis  
• OCR-based number plate reading  

Deployment:
• YOLO-based ANPR  
• Multi-vehicle tracking  
• Live CCTV streams  
""")
