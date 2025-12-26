import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
from fpdf import FPDF
import pandas as pd
import random
import requests
import io
import google.generativeai as genai

# -------------------------------------------------
# API CONFIGURATIONS
# -------------------------------------------------
OCR_API_KEY = "K85156107588957"
GEMINI_API_KEY = "AIzaSyDiRdTDdu8Uf1Tayy803DMkHCNnIj6B9DM"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Using stable flash model

# -------------------------------------------------
# SAFE YOLO IMPORT
# -------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Intelligent Vehicle Emission Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# STYLING
# -------------------------------------------------
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color:#0b1220; color:white; }
    .header { font-size:42px; font-weight:800; color: #00d4ff; }
    .sub { font-size:18px; opacity:0.8; margin-bottom: 20px; }
    .card {
        background:#111827;
        padding:25px;
        border-radius:16px;
        border: 1px solid #1f2937;
        margin-bottom:20px;
    }
    .bad { background:#3f1d1d; padding:15px; border-radius:10px; border-left: 5px solid #ff4b4b; margin-top:10px; }
    .good { background:#1d3f2a; padding:15px; border-radius:10px; border-left: 5px solid #28a745; margin-top:10px; }
    .metric-box { text-align: center; padding: 10px; background: #1f2937; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HELPER FUNCTIONS (OCR & GEMINI)
# -------------------------------------------------

def get_license_plate_ocr(image):
    """Extract text from license plate using OCR.space API"""
    try:
        # Convert PIL image to bytes
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        payload = {
            'apikey': OCR_API_KEY,
            'language': 'eng',
            'isOverlayRequired': False,
            'FileType': 'JPG',
        }
        files = {'file': ('image.jpg', byte_im, 'image/jpeg')}
        response = requests.post('https://api.ocr.space/parse/image', files=files, data=payload)
        result = response.json()

        if result.get("ParsedResults"):
            text = result["ParsedResults"][0]["ParsedText"].strip()
            # Clean text to look like a plate (remove special chars)
            clean_text = "".join(filter(str.isalnum, text))
            return clean_text if clean_text else "NOT_FOUND"
        return "OCR_ERROR"
    except Exception as e:
        return f"ERROR: {str(e)[:10]}"

def get_vehicle_analysis_gemini(image):
    """Analyze vehicle features and color using Gemini AI"""
    try:
        prompt = "Identify this vehicle's color, make/model (if visible), and specific physical features (e.g., roof rails, stickers, spoiler). Format: Color: [color], Features: [features]"
        response = gemini_model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return "Analysis unavailable: API limit or error."

@st.cache_resource
def load_yolo_model():
    if YOLO_AVAILABLE:
        try: return YOLO("yolov8n.pt")
        except: return None
    return None

def detect_smoke(image):
    img = np.array(image.convert("RGB")).astype(np.float32)
    v = np.max(img, axis=2)
    s = (v - np.min(img, axis=2)) / (v + 1e-6)
    smoke_mask = (v < 120) & (s < 0.45)
    score = np.sum(smoke_mask) / smoke_mask.size
    
    if score > 0.28: return score, "High"
    elif score > 0.15: return score, "Moderate"
    else: return score, "Low"

def generate_violation_id():
    return "VIO-" + datetime.now().strftime("%Y%m%d%H%M%S")

def generate_challan(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "GOVERNMENT OF INDIA - E-CHALLAN SYSTEM", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Electronic Pollution Violation Record", ln=True, align='C')
    pdf.ln(10)
    pdf.line(10, 30, 200, 30)
    
    for k, v in data.items():
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(50, 8, f"{k}:", ln=False)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, f"{v}")
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 5, "Note: This is an AI-generated notice. Please visit the RTO office or the official portal to pay the fine within 15 days.")
    
    return pdf.output(dest="S").encode("latin-1")

# -------------------------------------------------
# UI STRUCTURE
# -------------------------------------------------
st.markdown("<div class='header'>Intelligent Vehicle Emission Monitoring</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>AI-powered smoke detection, OCR plate reading & Gemini vehicle analysis</div>", unsafe_allow_html=True)

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Detection", "e-Challan", "Dashboard", "About"])

if "violation" not in st.session_state:
    st.session_state.violation = None

# -------------------------------------------------
# DETECTION PAGE
# -------------------------------------------------
if page == "Detection":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Feed")
        uploaded = st.file_uploader("Upload Vehicle Image (CCTV Frame)", type=["jpg", "png", "jpeg"])
        
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    if uploaded:
        with col2:
            st.subheader("AI Analysis Results")
            with st.spinner("Processing OCR and Gemini Analysis..."):
                # 1. Smoke Detection (Internal)
                smoke_score, severity = detect_smoke(image)
                
                # 2. OCR for Number Plate (External API)
                plate_number = get_license_plate_ocr(image)
                
                # 3. Gemini Analysis (External API)
                gemini_desc = get_vehicle_analysis_gemini(image)
                
                # 4. YOLO (Local Model)
                model = load_yolo_model()
                v_type = "Vehicle"
                if model:
                    results = model(image, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            label = model.names[int(box.cls[0])]
                            if label in ["car", "motorcycle", "bus", "truck"]:
                                v_type = label.capitalize()

                # Display Results
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write(f"🏷️ **Vehicle Type:** {v_type}")
                st.write(f"🔢 **Extracted Plate (OCR):** `{plate_number}`")
                st.write(f"💨 **Smoke Score:** {smoke_score:.2f} ({severity})")
                st.write(f"🎨 **Gemini AI Analysis:** \n{gemini_desc}")
                st.markdown("</div>", unsafe_allow_html=True)

                if severity == "High":
                    st.markdown("<div class='bad'>🚨 POLLUTING VEHICLE DETECTED - Violation Record Created</div>", unsafe_allow_html=True)
                    st.session_state.violation = {
                        "Violation ID": generate_violation_id(),
                        "Vehicle Number": plate_number,
                        "Vehicle Type": v_type,
                        "AI Vehicle Description": gemini_desc,
                        "Smoke Severity": severity,
                        "Smoke Score": round(smoke_score, 2),
                        "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Penalty": "Rs. 5000"
                    }
                else:
                    st.markdown("<div class='good'>✅ EMISSION WITHIN PERMISSIBLE LIMIT</div>", unsafe_allow_html=True)

# -------------------------------------------------
# E-CHALLAN PAGE
# -------------------------------------------------
elif page == "e-Challan":
    if st.session_state.violation:
        st.subheader("Auto Generated e-Challan Record")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        for k, v in st.session_state.violation.items():
            st.write(f"**{k}:** {v}")
        st.markdown("</div>", unsafe_allow_html=True)

        pdf = generate_challan(st.session_state.violation)
        st.download_button(
            "Download e-Challan PDF",
            pdf,
            f"challan_{st.session_state.violation['Vehicle Number']}.pdf",
            "application/pdf"
        )
    else:
        st.info("No violation detected yet. Please go to the Detection page.")

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
elif page == "Dashboard":
    st.subheader("Real-time City-wide Statistics")
    df = pd.DataFrame({
        "City": ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata"],
        "Violations Today": [142, 89, 76, 110, 65]
    })
    st.bar_chart(df.set_index("City"))
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Detections", "12,405", "+12%")
    c2.metric("Violations Found", "1,204", "5.4%")
    c3.metric("Revenue Generated", "₹6.02L", "+8%")

# -------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------
else:
    st.write("## 🚗 Intelligent Vehicle Emission Monitoring System")
    st.write("An advanced AI prototype for detecting smoke-emitting vehicles using OCR.space for plates and Gemini 2.0 for feature identification.")
    
    st.markdown("""
    ### 🏆 Hackathon Details
    - **Hackathon Name:** TechSprint 
    - **Team Name:** BLACK-DRAGON
    - **Team Leader:** Harsha (23BCE8747)
    - **Team Members:** Hasika (23BCE9934), Cheritha (23BCE7686)
    
    ### 🧠 Technology Stack
    - **Gemini 2.0 Flash:** Multi-modal analysis for vehicle features and color.
    - **OCR.space API:** High-speed License Plate Recognition.
    - **YOLOv8:** Object classification.
    - **Custom Image Processing:** Smoke density calculation.
    """)
    st.markdown("<hr style='border:1px solid #333;'><center><b>BLACK-DRAGON</b> | TechSprint Hackathon</center>", unsafe_allow_html=True)
