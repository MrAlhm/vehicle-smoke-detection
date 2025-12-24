import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import easyocr
import pandas as pd
from fpdf import FPDF
import tempfile
import re
import uuid

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Smart Vehicle Smoke Detection", layout="wide")
reader = easyocr.Reader(['en'], gpu=False)

# ===============================
# UI HEADER
# ===============================
st.markdown("""
<div style="padding:30px;border-radius:20px;
background:linear-gradient(120deg,#0f172a,#020617)">
<h1 style="color:white;font-size:40px;">Smart Vehicle Smoke Detection System</h1>
<p style="color:#cbd5f5;font-size:18px;">
AI-assisted pollution monitoring using CCTV images & videos
</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# FILE UPLOAD
# ===============================
uploaded = st.file_uploader(
    "Upload CCTV Image or Video",
    type=["jpg", "jpeg", "png", "mp4"]
)

# ===============================
# CORE LOGIC
# ===============================
def classify_vehicle(w, h):
    ratio = w / h
    if ratio > 2.8:
        return "Bus / Truck"
    elif ratio > 1.6:
        return "Car"
    else:
        return "Two-Wheeler"

def detect_smoke(vehicle_roi):
    hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    smoke_mask = (s < 70) & (v > 140)
    smoke_ratio = np.sum(smoke_mask) / smoke_mask.size

    if smoke_ratio > 0.35:
        return "High", smoke_ratio
    elif smoke_ratio > 0.20:
        return "Medium", smoke_ratio
    else:
        return "Low", smoke_ratio

def detect_plate(vehicle_roi):
    gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)

    for res in result:
        text = re.sub(r'[^A-Z0-9]', '', res[1].upper())
        if 6 <= len(text) <= 12:
            return text
    return "Not Readable"

def generate_challan(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "GOVERNMENT OF INDIA", ln=True, align="C")
    pdf.cell(0, 10, "Electronic Traffic Violation Notice", ln=True, align="C")
    pdf.ln(10)

    for k, v in data.items():
        pdf.cell(0, 10, f"{k}: {v}", ln=True)

    filename = f"e_challan_{data['Violation ID']}.pdf"
    pdf.output(filename)
    return filename

# ===============================
# IMAGE PROCESSING
# ===============================
def process_frame(frame, store_results=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < 8000:
            continue

        roi = frame[y:y+h, x:x+w]
        severity, score = detect_smoke(roi)

        if severity != "Low":
            plate = detect_plate(roi)
            vtype = classify_vehicle(w, h)

            results.append({
                "Vehicle Type": vtype,
                "Number Plate": plate,
                "Smoke Severity": severity,
                "Smoke Score": round(score,2),
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Violation ID": str(uuid.uuid4())[:8]
            })

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,f"{vtype} | {severity}",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    return frame, results

# ===============================
# MAIN
# ===============================
if uploaded:

    violations = []

    if uploaded.type.startswith("image"):
        img = np.array(Image.open(uploaded).convert("RGB"))
        frame, results = process_frame(img)

        st.image(frame, caption="Processed Frame", use_column_width=True)
        violations.extend(results)

    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, results = process_frame(frame)
            violations.extend(results)
            stframe.image(frame, channels="BGR")

        cap.release()

    # ===============================
    # RESULTS
    # ===============================
    if violations:
        st.subheader("Detected Violations")
        df = pd.DataFrame(violations)
        st.dataframe(df)

        last = violations[-1]
        st.subheader("Auto e-Challan")

        for k,v in last.items():
            st.write(f"**{k}:** {v}")

        pdf = generate_challan(last)
        with open(pdf,"rb") as f:
            st.download_button("Download e-Challan", f, pdf)

    else:
        st.success("No polluting vehicles detected")
