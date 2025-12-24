import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
from io import BytesIO
from fpdf import FPDF
import random
import pandas as pd

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Intelligent Vehicle Emission Monitoring",
    layout="wide"
)

# -------------------------------------------------
# STYLING
# -------------------------------------------------
st.markdown("""
<style>
body { background-color:#0b1220; color:white; }
.header { font-size:42px; font-weight:800; }
.sub { font-size:18px; opacity:0.8; }
.card {
    background:#111827;
    padding:20px;
    border-radius:16px;
    margin-bottom:20px;
}
.bad { background:#3f1d1d; padding:12px; border-radius:10px }
.good { background:#1d3f2a; padding:12px; border-radius:10px }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>Intelligent Vehicle Emission Monitoring</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>AI-powered smoke detection & automated pollution enforcement</div><br>", unsafe_allow_html=True)

# -------------------------------------------------
# BACKEND API SIMULATION (YOLO + ANPR)
# -------------------------------------------------
def backend_api_simulation():
    return {
        "vehicle_type": random.choice(["Car", "Truck", "Bus"]),
        "number_plate": random.choice([
            "DL8CAF5031",
            "TN09CB4455",
            "MH20DV2363",
            "KA01AB1234"
        ]),
        "confidence": random.randint(88, 97)
    }

# -------------------------------------------------
# SMOKE DETECTION (BLACK SMOKE LOGIC)
# -------------------------------------------------
def detect_smoke(image):
    img = np.array(image.convert("RGB")).astype(np.float32)
    v = np.max(img, axis=2)
    s = (v - np.min(img, axis=2)) / (v + 1e-6)

    smoke_mask = (v < 120) & (s < 0.45)
    score = np.sum(smoke_mask) / smoke_mask.size

    if score > 0.28:
        return score, "High"
    elif score > 0.15:
        return score, "Moderate"
    else:
        return score, "Low"

# -------------------------------------------------
# PDF GENERATOR
# -------------------------------------------------
def generate_challan(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0,10,"Government of India",ln=True)
    pdf.cell(0,10,"Electronic Pollution Violation Challan",ln=True)
    pdf.ln(5)

    for k,v in data.items():
        line = f"{k}: {v}".encode("latin-1","ignore").decode("latin-1")
        pdf.multi_cell(0,8,line)

    return pdf.output(dest="S").encode("latin-1")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Detection", "e-Challan", "Dashboard", "About"])

# -------------------------------------------------
# SESSION
# -------------------------------------------------
if "violation" not in st.session_state:
    st.session_state.violation = None

# -------------------------------------------------
# DETECTION PAGE
# -------------------------------------------------
if page == "Detection":
    uploaded = st.file_uploader("Upload Vehicle Image (CCTV Frame)", type=["jpg","png","jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_column_width=True)

        smoke_score, severity = detect_smoke(image)
        api_data = backend_api_simulation()

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"Smoke Score: **{smoke_score:.2f}**")
        st.write(f"Severity: **{severity}**")
        st.write(f"Vehicle Type: **{api_data['vehicle_type']}**")
        st.write(f"Number Plate: **{api_data['number_plate']}**")
        st.write(f"AI Confidence: **{api_data['confidence']}%**")
        st.markdown("</div>", unsafe_allow_html=True)

        if severity == "High":
            st.markdown("<div class='bad'>Polluting Vehicle Detected</div>", unsafe_allow_html=True)
            st.session_state.violation = {
                "Vehicle Number": api_data["number_plate"],
                "Vehicle Type": api_data["vehicle_type"],
                "Smoke Severity": severity,
                "Smoke Score": round(smoke_score,2),
                "AI Confidence": f"{api_data['confidence']}%",
                "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Penalty": "Rs. 5000"
            }
        else:
            st.markdown("<div class='good'>Emission Within Permissible Limit</div>", unsafe_allow_html=True)

# -------------------------------------------------
# E-CHALLAN PAGE
# -------------------------------------------------
elif page == "e-Challan":
    if st.session_state.violation:
        st.subheader("Auto Generated e-Challan")

        for k,v in st.session_state.violation.items():
            st.write(f"**{k}:** {v}")

        pdf = generate_challan(st.session_state.violation)

        st.download_button(
            "Download e-Challan PDF",
            pdf,
            "e_challan.pdf",
            "application/pdf"
        )
    else:
        st.info("No violation detected yet.")

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
elif page == "Dashboard":
    df = pd.DataFrame({
        "City": ["Delhi","Mumbai","Bengaluru","Chennai"],
        "Violations": [34,21,18,27]
    })
    st.bar_chart(df.set_index("City"))

# -------------------------------------------------
# ABOUT
# -------------------------------------------------
else:
    st.write("""
This prototype demonstrates an intelligent vehicle emission monitoring system.

- Smoke detection runs in real-time
- Vehicle and number plate detection run on edge AI servers in production
- Streamlit Cloud is used for visualization and enforcement workflow
- Scalable to city-wide deployment
""")
