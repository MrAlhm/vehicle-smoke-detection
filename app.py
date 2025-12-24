import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
from io import BytesIO
from fpdf import FPDF
import pytesseract

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Vehicle Emission Monitoring",
    layout="wide"
)

# -------------------------------------------------
# HEADER UI
# -------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}
.sub-title {
    font-size: 18px;
    opacity: 0.85;
}
.card {
    background-color: #0e1a2b;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Smart Vehicle Emission Monitoring</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered vehicle smoke detection & enforcement prototype</div><br>", unsafe_allow_html=True)

# -------------------------------------------------
# SMOKE DETECTION (BLACK SMOKE FIXED)
# -------------------------------------------------
def detect_smoke(image):
    img = np.array(image.convert("RGB")).astype(np.float32)

    # HSV-like logic without OpenCV
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    v = np.max(img, axis=2)
    s = (v - np.min(img, axis=2)) / (v + 1e-6)

    # BLACK smoke characteristics
    smoke_mask = (v < 120) & (s < 0.45)
    smoke_score = np.sum(smoke_mask) / smoke_mask.size

    if smoke_score > 0.28:
        severity = "High"
    elif smoke_score > 0.15:
        severity = "Moderate"
    else:
        severity = "Low"

    return smoke_score, severity

# -------------------------------------------------
# VEHICLE TYPE (SAFE DEMO LOGIC)
# -------------------------------------------------
def detect_vehicle_type(image):
    w, h = image.size
    if w > 900:
        return "Truck / Bus"
    elif w > 600:
        return "Car"
    else:
        return "Two-Wheeler"

# -------------------------------------------------
# NUMBER PLATE OCR (SAFE CLOUD VERSION)
# -------------------------------------------------
def detect_number_plate(image):
    gray = image.convert("L")
    text = pytesseract.image_to_string(gray, config="--psm 6")
    text = text.replace("\n", "").replace(" ", "")
    if len(text) < 6:
        return "Not Readable"
    return text[:12]

# -------------------------------------------------
# E-CHALLAN PDF (UNICODE SAFE)
# -------------------------------------------------
def generate_challan(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Electronic Traffic Violation Notice", ln=True)
    pdf.ln(5)

    for k, v in data.items():
        safe_line = f"{k}: {str(v)}"
        safe_line = safe_line.encode("latin-1", "ignore").decode("latin-1")
        pdf.multi_cell(0, 8, safe_line)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Vehicle Frame", use_column_width=True)

    smoke_score, severity = detect_smoke(image)
    vehicle_type = detect_vehicle_type(image)
    plate = detect_number_plate(image)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    col1.metric("Smoke Score", f"{smoke_score:.2f}")
    col2.metric("Severity", severity)
    col3.metric("Vehicle Type", vehicle_type)

    st.markdown("<br>", unsafe_allow_html=True)

    if severity == "High":
        st.error("🚨 Polluting Vehicle Detected")
    elif severity == "Moderate":
        st.warning("⚠️ Moderate Emission Detected")
    else:
        st.success("✅ Emission Within Permissible Limit")

    st.markdown("### Vehicle Identification")
    st.info(f"Detected Number Plate: **{plate}**")

    # -------------------------------------------------
    # E-CHALLAN
    # -------------------------------------------------
    if severity == "High":
        challan_data = {
            "Violation Type": "Excessive Vehicular Emission",
            "Vehicle Number": plate,
            "Vehicle Type": vehicle_type,
            "Smoke Severity": severity,
            "Smoke Score": round(smoke_score, 2),
            "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Fine Amount": "₹1000"
        }

        pdf = generate_challan(challan_data)

        st.download_button(
            "⬇ Download e-Challan (PDF)",
            data=pdf,
            file_name="e_challan.pdf",
            mime="application/pdf"
        )

else:
    st.info("Please upload a vehicle image to start detection.")
