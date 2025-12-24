import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import easyocr
import tempfile
import matplotlib.pyplot as plt
from fpdf import FPDF

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart City Vehicle Smoke Monitoring",
    layout="wide",
    page_icon="🚦"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
body { background-color: #f5f7fb; }
.hero {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(90deg,#1e3c72,#2a5298);
    color: white;
}
.card {
    padding: 20px;
    border-radius: 16px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    text-align: center;
}
.badge-red { background:#e74c3c;color:white;padding:10px 20px;border-radius:20px; }
.badge-green { background:#2ecc71;color:white;padding:10px 20px;border-radius:20px; }
.badge-yellow { background:#f1c40f;color:black;padding:10px 20px;border-radius:20px; }
</style>
""", unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<div class="hero">
<h1>🚗 Smart City Vehicle Smoke Monitoring</h1>
<p>AI-Powered Real-Time Pollution Detection & Enforcement System</p>
</div>
""", unsafe_allow_html=True)

# ================= OCR =================
reader = easyocr.Reader(['en'], gpu=False)

# ================= SESSION =================
if "violations" not in st.session_state:
    st.session_state.violations = []

# ================= FUNCTIONS =================
def detect_smoke(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = (s < 60) & (v > 150)
    score = np.sum(mask) / mask.size
    if score >= 0.30:
        return score, "High", "🚨 Violation"
    elif score >= 0.20:
        return score, "Medium", "⚠️ Warning"
    else:
        return score, "Low", "✅ Clean"

def detect_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)
    cnts,_ = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        approx = cv2.approxPolyDP(c,0.02*cv2.arcLength(c,True),True)
        if len(approx)==4:
            x,y,w,h = cv2.boundingRect(approx)
            plate = img[y:y+h,x:x+w]
            txt = reader.readtext(plate)
            return txt[0][1] if txt else "Not Readable", plate
    return "Not Readable", None

def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200,10,"E-CHALLAN – VEHICLE EMISSION VIOLATION",ln=True,align="C")
    pdf.ln(10)
    for k,v in data.items():
        pdf.cell(200,10,f"{k}: {v}",ln=True)
    path = f"/tmp/challan_{datetime.now().timestamp()}.pdf"
    pdf.output(path)
    return path

# ================= SIDEBAR =================
st.sidebar.title("📂 Navigation")
section = st.sidebar.radio(
    "Go to",
    ["🔍 Detection","🎥 CCTV","🧾 e-Challan","📊 Dashboard","🗺 Hotspots","ℹ️ About"]
)

camera = st.sidebar.selectbox(
    "📍 Camera Location",
    ["Cam-Delhi-01","Cam-Delhi-02","Cam-Delhi-03"]
)

# ================= DETECTION =================
if section == "🔍 Detection":
    file = st.file_uploader("Upload Vehicle Image",["jpg","png","jpeg"])
    if file:
        img = Image.open(file)
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np,cv2.COLOR_RGB2BGR)

        score, severity, status = detect_smoke(img_bgr)
        plate, plate_img = detect_plate(img_bgr)

        c1,c2,c3 = st.columns(3)
        c1.markdown(f"<div class='card'><h3>Smoke Score</h3><h2>{score:.2f}</h2></div>",unsafe_allow_html=True)
        c2.markdown(f"<div class='card'><h3>Severity</h3><h2>{severity}</h2></div>",unsafe_allow_html=True)
        if "Violation" in status:
            c3.markdown("<div class='badge-red'>VIOLATION</div>",unsafe_allow_html=True)
        elif "Warning" in status:
            c3.markdown("<div class='badge-yellow'>WARNING</div>",unsafe_allow_html=True)
        else:
            c3.markdown("<div class='badge-green'>CLEAN</div>",unsafe_allow_html=True)

        st.image(img,caption="Captured Frame",use_column_width=True)
        st.info(f"Number Plate: {plate}")

        if "Violation" in status:
            data = {
                "Plate":plate,
                "Camera":camera,
                "Severity":severity,
                "Time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.violations.append(data)
            st.error("🚨 Auto e-Challan Generated")

# ================= CCTV =================
elif section == "🎥 CCTV":
    video = st.file_uploader("Upload CCTV Video",["mp4","avi","mov"])
    if video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())
        cap = cv2.VideoCapture(tfile.name)
        frames = 0
        while cap.isOpened() and frames < 20:
            ret, frame = cap.read()
            if not ret: break
            score, severity, status = detect_smoke(frame)
            if "Violation" in status:
                st.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                         caption=f"Frame {frames+1} – Smoke Detected")
            frames += 1
        st.success("CCTV Analysis Completed")

# ================= CHALLAN =================
elif section == "🧾 e-Challan":
    if st.session_state.violations:
        df = pd.DataFrame(st.session_state.violations)
        st.dataframe(df,use_container_width=True)
        latest = st.session_state.violations[-1]
        pdf = generate_pdf(latest)
        with open(pdf,"rb") as f:
            st.download_button("⬇ Download Latest e-Challan PDF",f,"e_challan.pdf")
    else:
        st.info("No challans yet")

# ================= DASHBOARD =================
elif section == "📊 Dashboard":
    if st.session_state.violations:
        df = pd.DataFrame(st.session_state.violations)
        st.bar_chart(df["Camera"].value_counts())
        st.line_chart(df.groupby("Camera").size())
    else:
        st.info("No data available")

# ================= HOTSPOTS =================
elif section == "🗺 Hotspots":
    st.markdown("### 📍 Pollution Hotspots (Simulated Map)")
    st.write("🔴 Red = High violation zones")
    st.map(pd.DataFrame({
        "lat":[28.61,28.62,28.63],
        "lon":[77.20,77.21,77.22]
    }))

# ================= ABOUT =================
else:
    st.markdown("""
### 🚦 Smart City Vehicle Smoke Monitoring
**Purpose**  
Automated detection and enforcement of vehicular emission violations.

**Tech Stack**
- Computer Vision
- EasyOCR
- OpenCV
- Streamlit
- AI-based Heuristics

**Impact**
- Reduced pollution
- Real-time enforcement
- Data-driven policy making
""")

st.caption("© Smart City AI – Hackathon Prototype")
