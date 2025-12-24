import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import easyocr
import pandas as pd
from fpdf import FPDF
import io

# ------------------------------
# App Config
# ------------------------------
st.set_page_config(
    page_title="Smart Vehicle Emission Monitoring",
    layout="wide"
)

# ------------------------------
# Styling (Professional, Clean)
# ------------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.big-title {
    font-size: 42px;
    font-weight: 700;
}
.sub-title {
    font-size: 18px;
    color: #cbd5e1;
}
.card {
    background-color: #020617;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 20px;
}
.bad {
    background-color: #3f1d1d;
    padding: 12px;
    border-radius: 10px;
}
.good {
    background-color: #1d3f2a;
    padding: 12px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Header
# ------------------------------
st.markdown(
    '<div class="card">'
    '<div class="big-title">Smart Vehicle Emission Monitoring</div>'
    '<div class="sub-title">AI-powered pollution detection & enforcement platform</div>'
    '</div>',
    unsafe_allow_html=True
)

# ------------------------------
# Initialize OCR (Deep Learning)
# ------------------------------
reader = easyocr.Reader(['en'], gpu=False)

# ------------------------------
# Smoke Detection (PIL + NumPy)
# ------------------------------
def detect_smoke_pil(image):
    img = np.array(image.convert("RGB"))
    gray = np.mean(img, axis=2)

    bright_pixels = gray > 200
    smoke_score = np.sum(bright_pixels) / bright_pixels.size

    if smoke_score > 0.28:
        return smoke_score, "High"
    elif smoke_score > 0.18:
        return smoke_score, "Moderate"
    else:
        return smoke_score, "Low"

# ------------------------------
# Number Plate OCR
# ------------------------------
def detect_number_plate(image):
    results = reader.readtext(np.array(image))
    for (_, text, conf) in results:
        if len(text) >= 6 and conf > 0.4:
            return text.upper()
    return "Not Readable"

# ------------------------------
# Vehicle Type (Rule-Based)
# ------------------------------
def detect_vehicle_type(image):
    w, h = image.size
    ratio = w / h

    if ratio > 2.2:
        return "Bus / Truck"
    elif ratio > 1.5:
        return "Car / SUV"
    else:
        return "Two-Wheeler"

# ------------------------------
# e-Challan PDF
# ------------------------------
def generate_challan(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Government of India", ln=True)
    pdf.cell(0, 10, "Auto-Generated Pollution e-Challan", ln=True)
    pdf.ln(5)

    for k, v in data.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# ------------------------------
# Upload Section
# ------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload Vehicle Image (CCTV Frame Simulation)",
    type=["jpg", "jpeg", "png"]
)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Main Processing
# ------------------------------
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Captured Vehicle Frame", use_column_width=True)

    smoke_score, severity = detect_smoke_pil(image)
    plate = detect_number_plate(image)
    vehicle_type = detect_vehicle_type(image)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Detection Result")

    st.write(f"Smoke Score: **{smoke_score:.2f}**")
    st.write(f"Smoke Severity: **{severity}**")

    if severity == "High":
        st.markdown('<div class="bad">🚨 Polluting Vehicle Detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="good">✅ Emission Within Limit</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Vehicle Identification")
    st.write(f"Number Plate: **{plate}**")
    st.write(f"Vehicle Type: **{vehicle_type}**")
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------
    # e-Challan
    # ------------------------------
    if severity == "High":
        challan_data = {
            "Number Plate": plate,
            "Vehicle Type": vehicle_type,
            "Smoke Severity": severity,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Fine Amount": "₹2000",
            "Violation": "Excessive Vehicular Emission"
        }

        pdf = generate_challan(challan_data)

        st.download_button(
            "Download e-Challan (PDF)",
            data=pdf,
            file_name="e_challan.pdf",
            mime="application/pdf"
        )

# ------------------------------
# Dashboard (History)
# ------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Violation Analytics (Demo Data)")

df = pd.DataFrame({
    "City": ["Delhi", "Delhi", "Mumbai", "Delhi", "Bengaluru"],
    "Severity": ["High", "Moderate", "High", "High", "Moderate"]
})

st.bar_chart(df["City"].value_counts())
st.markdown('</div>', unsafe_allow_html=True)
