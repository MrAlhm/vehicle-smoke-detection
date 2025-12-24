import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
from fpdf import FPDF
import io
import random
import pandas as pd

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Vehicle Emission Monitoring",
    layout="wide"
)

# --------------------------------------------------
# UI STYLES
# --------------------------------------------------
st.markdown("""
<style>
body { background-color:#0b1220; color:white; }
.card { background:#111827; padding:25px; border-radius:16px; margin-bottom:20px }
.big { font-size:44px; font-weight:800 }
.sub { font-size:18px; opacity:0.8 }
.bad { background:#3f1d1d; padding:12px; border-radius:10px }
.good { background:#1d3f2a; padding:12px; border-radius:10px }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="card">
  <div class="big">Smart Vehicle Emission Monitoring</div>
  <div class="sub">AI-based smoke & emission violation detection system</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STORAGE
# --------------------------------------------------
if "last_violation" not in st.session_state:
    st.session_state.last_violation = None

# --------------------------------------------------
# YOLO INTERFACE (CLOUD SAFE)
# --------------------------------------------------
class YOLODetector:
    """
    Interface compatible with real YOLO.
    On Streamlit Cloud, this runs in simulation mode.
    """
    def detect(self, image):
        return {
            "vehicle_type": random.choice(["Car", "Bus", "Truck"]),
            "number_plate": random.choice([
                "MH20DV2363",
                "DL8CAF5031",
                "KA01AB1234",
                "TN09CB4455"
            ]),
            "bbox": [120, 80, 520, 340]  # simulated bounding box
        }

yolo = YOLODetector()

# --------------------------------------------------
# SMOKE DETECTION (STABLE)
# --------------------------------------------------
def detect_smoke(image):
    arr = np.array(image.convert("RGB"))
    gray = np.mean(arr, axis=2)
    smoke_ratio = np.sum(gray > 200) / gray.size

    if smoke_ratio > 0.30:
        return smoke_ratio, "High"
    elif smoke_ratio > 0.18:
        return smoke_ratio, "Moderate"
    else:
        return smoke_ratio, "Low"

# --------------------------------------------------
# PDF GENERATION (UNICODE SAFE)
# --------------------------------------------------
def generate_challan(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0,10,"Government of India",ln=True)
    pdf.cell(0,10,"Electronic Pollution Violation Challan",ln=True)
    pdf.ln(5)

    for k,v in data.items():
        safe = f"{k}: {v}".encode("latin-1","ignore").decode("latin-1")
        pdf.multi_cell(0,8,safe)

    return pdf.output(dest="S").encode("latin-1")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Detection", "e-Challan", "Dashboard", "About"]
)

# --------------------------------------------------
# DETECTION PAGE
# --------------------------------------------------
if page == "Detection":

    uploaded = st.file_uploader(
        "Upload Vehicle Image (CCTV Frame Simulation)",
        type=["jpg","jpeg","png"]
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_column_width=True)

        # Smoke detection
        smoke_score, severity = detect_smoke(image)

        # YOLO (simulated)
        yolo_result = yolo.detect(image)

        vehicle_type = yolo_result["vehicle_type"]
        number_plate = yolo_result["number_plate"]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"Smoke Score: **{smoke_score:.2f}**")
        st.write(f"Severity: **{severity}**")
        st.write(f"Vehicle Type: **{vehicle_type}**")
        st.write(f"Number Plate: **{number_plate}**")
        st.markdown("</div>", unsafe_allow_html=True)

        if severity == "High":
            st.markdown("<div class='bad'>Polluting Vehicle Detected</div>", unsafe_allow_html=True)

            st.session_state.last_violation = {
                "Vehicle Number": number_plate,
                "Vehicle Type": vehicle_type,
                "Smoke Severity": severity,
                "Smoke Score": round(smoke_score,2),
                "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Penalty": "Rs. 5000"
            }
        else:
            st.markdown("<div class='good'>Emission Within Permissible Limit</div>", unsafe_allow_html=True)

# --------------------------------------------------
# E-CHALLAN PAGE
# --------------------------------------------------
elif page == "e-Challan":

    if st.session_state.last_violation:
        st.subheader("Auto Generated e-Challan")

        for k,v in st.session_state.last_violation.items():
            st.write(f"**{k}:** {v}")

        pdf_bytes = generate_challan(st.session_state.last_violation)

        st.download_button(
            "Download e-Challan PDF",
            data=pdf_bytes,
            file_name="e_challan.pdf",
            mime="application/pdf"
        )
    else:
        st.info("No violation detected yet.")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
elif page == "Dashboard":
    data = {
        "City": ["Delhi","Mumbai","Bengaluru","Chennai"],
        "Violations": [34,21,18,27]
    }
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index("City"))

# --------------------------------------------------
# ABOUT
# --------------------------------------------------
else:
    st.write("""
    This prototype demonstrates a cloud-safe AI system for vehicle emission monitoring.

    - YOLO-based vehicle & number plate detection (edge deployment)
    - Smoke severity analysis
    - Auto-generated pollution challans
    - Scalable architecture for real CCTV feeds
    """)
