import os
import io
import base64
import hashlib
import re
from datetime import datetime

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --- 1. INITIALIZATION ---
st.set_page_config(page_title="PyParkOps", page_icon="pyparkops-logo.png", layout="centered")
HOURLY_RATE = 50  # Taka
DB_FILE = "parking_database.csv"
MAX_IMAGE_SIZE_BYTES = 2 * 1024 * 1024

def load_logo_base64():
    try:
        with open("pyparkops-logo.png", "rb") as logo_file:
            return base64.b64encode(logo_file.read()).decode("utf-8")
    except FileNotFoundError:
        return "" # Fallback if image is missing

LOGO_BASE64 = load_logo_base64()

def ensure_database():
    if not os.path.exists(DB_FILE) or os.stat(DB_FILE).st_size == 0:
        df = pd.DataFrame(columns=["Plate_Number", "Entry_Time"])
        df.to_csv(DB_FILE, index=False)

ensure_database()

@st.cache_resource
def load_models():
    yolo = YOLO("bd_plates_best_yolo26.pt")
    reader = easyocr.Reader(["bn", "en"], gpu=False)
    return yolo, reader

yolo_model, ocr_reader = load_models()

def load_active_db():
    ensure_database()
    return pd.read_csv(DB_FILE)

def parse_time_string(time_string):
    time_string = str(time_string)
    try:
        return datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        now = datetime.now()
        return datetime.strptime(time_string, "%H:%M:%S").replace(
            year=now.year, month=now.month, day=now.day
        )

def format_license_plate(raw_text):
    """Formats OCR text into: CityName [- মেট্রো] - Class  XX-XXXX"""
    # Normalize OCR separators while keeping the Bengali text intact.
    normalized_text = re.sub(r"[\s_\-]+", " ", str(raw_text)).strip()

    # Split the text prefix from the numeric suffix at the first digit.
    prefix_match = re.match(r"^(.*?)([0-9\u09E6-\u09EF].*)$", normalized_text)
    if not prefix_match:
        return normalized_text

    prefix_text = prefix_match.group(1).strip()

    # Extract the last 6 digits from the OCR output, allowing Bengali numerals too.
    digit_matches = re.findall(r"[0-9\u09E6-\u09EF]", prefix_match.group(2))
    if len(digit_matches) < 6:
        return normalized_text

    num_part = "".join(digit_matches[-6:])

    # Find the single-letter vehicle class token anywhere before the digits.
    prefix_tokens = prefix_text.split()
    class_index = None
    for index, token in enumerate(prefix_tokens):
        if len(token) == 1 and re.search(r"[\u0980-\u09FF]", token):
            class_index = index

    if class_index is not None:
        veh_class = prefix_tokens[class_index]
        city_tokens = [token for index, token in enumerate(prefix_tokens) if index != class_index]
        formatted_text = f"{' '.join(city_tokens)} - {veh_class}".strip()
    else:
        formatted_text = prefix_text

    formatted_num = f"{num_part[:2]}-{num_part[2:]}"
    return f"{formatted_text}  {formatted_num}".strip()

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
        .stApp { background-color: #12141A; color: #ffffff; }
        .header-container { display: flex; align-items: center; gap: 15px; margin-bottom: 20px; }
        .logo-image { width: 60px; height: 60px; border-radius: 12px; object-fit: cover; background: white; flex-shrink: 0; display: block; }
        .header-copy { display: flex; flex-direction: column; justify-content: center; }
        .title-text { font-size: 38px; font-weight: 800; line-height: 1; margin: 0; color: #ffffff; }
        .title-accent { color: #FF4B4B; }
        .subtitle-text { color: #d1d5db; font-size: 20px; margin-top: 5px; font-weight: 500; }
        .info-card { background-color: #ffffff; border-radius: 10px; padding: 20px; color: #000000; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 15px; height: 100px; display: flex; flex-direction: column; justify-content: center; }
        .card-label { font-size: 14px; font-weight: 600; color: #4b5563; margin-bottom: 5px; }
        .card-value { font-size: 24px; font-weight: 800; color: #111827; }
        .license-value { font-size: 30px; }
        .table-text { padding-top: 10px; font-size: 16px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. UI HELPER COMPONENTS ---
def draw_card(label, value, is_license=False):
    val_class = "license-value" if is_license else "card-value"
    return f"""
        <div class="info-card">
            <div class="card-label">{label}</div>
            <div class="{val_class}">{value}</div>
        </div>
    """

# --- 4. CORE LOGIC ---
def process_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(img_rgb)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_plate = img_rgb[y1:y2, x1:x2]
            ocr_results = ocr_reader.readtext(cropped_plate, detail=1)

            if ocr_results:
                all_text_parts = [detection[1] for detection in ocr_results if detection and len(detection) > 1]
                if all_text_parts:
                    raw_joined = " ".join(all_text_parts)
                    return format_license_plate(raw_joined) # Pass through regex formatter
    return None

def manage_parking(plate_number):
    df = load_active_db()
    now = datetime.now()
    parked_car = df[df["Plate_Number"] == plate_number]

    if parked_car.empty:
        new_row = pd.DataFrame({"Plate_Number": [plate_number], "Entry_Time": [now.strftime("%Y-%m-%d %H:%M:%S")]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DB_FILE, index=False)
        return "ENTRY", now, None, None

    entry_time_str = parked_car.iloc[0]["Entry_Time"]
    entry_time = parse_time_string(entry_time_str)
    
    duration = now - entry_time
    hours_parked = round(duration.total_seconds() / 3600, 4)
    total_fee = max(HOURLY_RATE, int(hours_parked * HOURLY_RATE))

    df = df[df["Plate_Number"] != plate_number]
    df.to_csv(DB_FILE, index=False)
    return "EXIT", entry_time, hours_parked, total_fee

# --- 5. APP LAYOUT ---
if "last_scan" not in st.session_state:
    st.session_state.last_scan = None
if "image_gallery" not in st.session_state:
    st.session_state.image_gallery = []

# Header
if LOGO_BASE64:
    img_tag = f'<img class="logo-image" src="data:image/png;base64,{LOGO_BASE64}" alt="PyParkOps logo" />'
else:
    img_tag = '<div class="logo-image" style="background:#FF4B4B; color:white; display:flex; align-items:center; justify-content:center; font-size:36px; font-weight:bold;">P</div>'

st.markdown(f"""
    <div class="header-container">
        {img_tag}
        <div class="header-copy">
            <h1 class="title-text">PyPark<span class="title-accent">Ops</span></h1>
            <div class="subtitle-text">A Car Parking Operations System</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Parse data for UI Cards
last = st.session_state.last_scan
detected_plate = last["plate_text"] if last else "—"

if last and last.get("fee"):
    payable = f"৳{last['fee']} <span style='font-size: 14px; font-weight: normal; color: #6b7280;'>(৳{HOURLY_RATE}/hr)</span>"
else:
    payable = f"৳{HOURLY_RATE}/hr"

in_time = last["entry_time"].strftime("%I:%M:%S %p") if last and last.get("entry_time") else "—"
out_time = last["scanned_at"].strftime("%I:%M:%S %p") if last and last.get("action") == "EXIT" else "—"
duration_hrs = f"{last['hours']}hrs" if last and last.get("hours") else "—"

# Top Row Cards
col1, col2 = st.columns([2.5, 1.5])
with col1:
    st.markdown(draw_card("License No. 🚗", detected_plate, is_license=True), unsafe_allow_html=True)
with col2:
    st.markdown(draw_card("Payable 💸", payable), unsafe_allow_html=True)

# Bottom Row Cards
col3, col4, col5 = st.columns(3)
with col3:
    st.markdown(draw_card("In Time 🕔", in_time), unsafe_allow_html=True)
with col4:
    st.markdown(draw_card("Out Time 🕔", out_time), unsafe_allow_html=True)
with col5:
    st.markdown(draw_card("Duration ⏳", duration_hrs), unsafe_allow_html=True)

st.write("---")

# Image Upload
uploaded_file = st.file_uploader("Scan Vehicle", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
        st.error("Image size must be 2MB or smaller. Please upload a smaller file.")
        st.stop()

    file_hash = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.get("last_file_hash") != file_hash:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        preview_img = pil_img.resize((300, 300))

        existing_hashes = {item["hash"] for item in st.session_state.image_gallery}
        if file_hash not in existing_hashes:
            st.session_state.image_gallery.append({
                "hash": file_hash, "image": preview_img, "name": uploaded_file.name
            })

        with st.spinner("Scanning..."):
            plate_text = process_image(uploaded_file)

            if plate_text:
                action, entry_time, hours, fee = manage_parking(plate_text)
                st.session_state.last_scan = {
                    "plate_text": plate_text, "action": action, "entry_time": entry_time,
                    "hours": hours, "fee": fee, "scanned_at": datetime.now(),
                }
            else:
                st.session_state.last_scan = None
                st.error("No license plate detected.")

        st.session_state.last_file_hash = file_hash
        st.rerun()

st.markdown('<div class="card-label">Uploaded Images</div>', unsafe_allow_html=True)
if st.session_state.image_gallery:
    for start_index in range(0, len(st.session_state.image_gallery), 3):
        row_items = st.session_state.image_gallery[start_index:start_index + 3]
        columns = st.columns(3)
        for column, item in zip(columns, row_items):
            with column:
                st.image(item["image"], use_container_width=True, caption=item["name"])
else:
    st.info("Uploaded images will appear here as 300 x 300 tiles, three per row.")

# --- 6. ACTIVE PARKING LIST ---
st.write("---")
st.subheader("📋 Active Parking Database")
active_db = load_active_db()

if active_db.empty:
    st.info("No active parked vehicles.")
else:
    h1, h2, h3, h4 = st.columns([2.5, 2, 1.5, 1])
    h1.markdown("**Plate Number**")
    h2.markdown("**Entry Time**")
    h3.markdown("**Duration**")
    h4.markdown("**Action**")
    st.markdown("<hr style='margin: 0px; opacity: 0.2;'/>", unsafe_allow_html=True)

    for idx, row in active_db.reset_index().iterrows():
        plate = row['Plate_Number']
        entry_time = parse_time_string(row['Entry_Time'])
        
        duration = datetime.now() - entry_time
        hours = round(duration.total_seconds() / 3600, 2)
        display_time = entry_time.strftime("%I:%M %p")

        r1, r2, r3, r4 = st.columns([2.5, 2, 1.5, 1])
        with r1:
            st.markdown(f"<div class='table-text'><b>{plate}</b></div>", unsafe_allow_html=True)
        with r2:
            st.markdown(f"<div class='table-text'>{display_time}</div>", unsafe_allow_html=True)
        with r3:
            st.markdown(f"<div class='table-text'>{hours} hrs</div>", unsafe_allow_html=True)
        with r4:
            if st.button("🗑️ Delete", key=f"del_{idx}"):
                df = load_active_db()
                df = df.drop(index=row['index'])
                df.to_csv(DB_FILE, index=False)
                st.rerun()