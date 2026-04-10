from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64

import os, json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

import easyocr
from datetime import datetime

app = FastAPI()

latest_frame = None
latest_result = None

# =========================
# MODELS
# =========================
vehicle_model = YOLO("yolov8m.pt")
helmet_model = YOLO("helmet_best.pt")
seatbelt_model = YOLO("seatbelt_best.pt")
plate_model = YOLO("plate_best.pt")

ocr_reader = easyocr.Reader(['en'], gpu=False)

# =========================
# GOOGLE SHEETS
# =========================
try:
    creds_dict = json.loads(os.environ["GOOGLE_CREDENTIALS"])
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open("Road Safety Violations").sheet1
except:
    sheet = None

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

TWO_WHEELER_IDS = {3}
FOUR_WHEELER_IDS = {2, 5, 7}

# =========================
# OCR
# =========================
def extract_plate_text(crop):
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        results = ocr_reader.readtext(gray)
        if results:
            return max(results, key=lambda x: x[2])[1].upper()
        return "N/A"
    except:
        return "N/A"

# =========================
# LOGGING
# =========================
def log_violation(detections):
    if sheet is None:
        return
    for d in detections:
        sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            d["vehicle_type"],
            d["violation"],
            d["plate"],
            d["confidence"],
            "violation"
        ])

# =========================
# DETECTION
# =========================
def run_detection(frame):

    detections = []

    results = vehicle_model(frame, imgsz=640, verbose=False)[0]

    if results.boxes is not None:

        boxes_sorted = sorted(
            results.boxes,
            key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]),
            reverse=True
        )

        for box in boxes_sorted:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in VEHICLE_CLASSES or conf < 0.6:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # filter small vehicles
            if (x2 - x1) < 120 or (y2 - y1) < 120:
                continue

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, VEHICLE_CLASSES[cls], (x1, y1-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            h, w, _ = frame.shape
            crop = frame[
                max(0, y1-60):min(h, y2+60),
                max(0, x1-60):min(w, x2+60)
            ]

            # =====================
            # PLATE OCR
            # =====================
            plate_text = "N/A"
            p_results = plate_model(crop, conf=0.5, verbose=False)[0]

            if p_results.boxes is not None:
                for pbox in p_results.boxes:
                    px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                    plate_crop = crop[py1:py2, px1:px2]
                    plate_text = extract_plate_text(plate_crop)
                    break

            # =====================
            # HELMET
            # =====================
            if cls in TWO_WHEELER_IDS:

                ch, cw, _ = crop.shape
                head_region = crop[0:int(ch*0.6), :]

                h_results = helmet_model(head_region, conf=0.15, verbose=False)[0]
                helmet_detected = False

                if h_results.boxes is not None:
                    for hbox in h_results.boxes:
                        h_conf = float(hbox.conf[0])
                        h_cls = int(hbox.cls[0])

                        # class 0 = helmet
                        if h_conf > 0.2 and h_cls == 0:
                            helmet_detected = True
                            break

                if not helmet_detected:
                    cv2.putText(frame, "NO HELMET", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                    detections.append({
                        "vehicle_type": VEHICLE_CLASSES[cls],
                        "violation":"NO_HELMET",
                        "confidence":0.5,
                        "plate":plate_text
                    })

            # =====================
            # SEATBELT
            # =====================
            elif cls in FOUR_WHEELER_IDS:

                if (x2 - x1) < 220:
                    continue

                ch, cw, _ = crop.shape
                driver = crop[int(ch*0.3):int(ch*0.75), int(cw*0.25):int(cw*0.65)]

                s_results = seatbelt_model(driver, conf=0.6, verbose=False)[0]

                seatbelt_found = False
                best_conf = 0.0

                if s_results.boxes is not None:
                    for sbox in s_results.boxes:
                        s_conf = float(sbox.conf[0])
                        best_conf = max(best_conf, s_conf)

                        if s_conf > 0.6 and int(sbox.cls[0]) == 0:
                            seatbelt_found = True
                            break

                if not seatbelt_found:
                    cv2.putText(frame, "NO SEATBELT", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                    detections.append({
                        "vehicle_type": VEHICLE_CLASSES[cls],
                        "violation":"NO_SEATBELT",
                        "confidence": round(best_conf, 2) if best_conf > 0 else 0.7,
                        "plate":plate_text
                    })

    # remove duplicates
    unique = []
    seen = set()

    for d in detections:
        key = (d["violation"], d["plate"])
        if key not in seen:
            seen.add(key)
            unique.append(d)

    return {
        "status":"violation" if unique else "safe",
        "violations":unique
    }

# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return HTMLResponse('<meta http-equiv="refresh" content="0; url=/ui">')

@app.get("/health")
def health():
    return {"status": "running"}

# =========================
# UI (UNCHANGED)
# =========================
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """<html>
    <head>
        <meta charset="UTF-8">
        <title>Road Safety Monitor</title>
        <style>
            * { box-sizing: border-box; font-family: 'Segoe UI', sans-serif; }
            body {
                margin: 0;
                height: 100vh;
                background: linear-gradient(135deg, #0f172a, #020617);
                display: flex;
                justify-content: center;
                align-items: center;
                color: white;
            }
            .container {
                width: 500px;
                background: #111827;
                padding: 40px;
                border-radius: 16px;
                box-shadow: 0 20px 50px rgba(0,0,0,0.6);
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .title { font-size: 24px; margin-bottom: 30px; }
            form { width: 100%; display: flex; flex-direction: column; align-items: center; }
            .upload-box {
                width: 100%;
                border: 2px dashed #38bdf8;
                border-radius: 12px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: 0.3s;
            }
            .upload-box:hover { background: rgba(56,189,248,0.1); }
            input[type="file"] { display: none; }
            button {
                margin-top: 20px;
                width: 100%;
                padding: 12px;
                background: #38bdf8;
                border: none;
                border-radius: 10px;
            }
            .link { margin-top: 20px; color: #38bdf8; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="title">🚦 Road Safety Monitor</div>
            <form action="/detect_ui" method="post" enctype="multipart/form-data">
                <div class="upload-box" onclick="document.getElementById('fileInput').click()">
                    Click or Drag Image Here
                </div>
                <input type="file" name="file" id="fileInput" onchange="previewImage(event)" hidden>
                <div id="file-name" style="margin-top:10px; color:#94a3b8;"></div>
                <img id="preview" style="margin-top:15px; max-width:100%; border-radius:10px; display:none;"/>
                <button type="submit">Run Detection</button>
            </form>
            <a href="/live" class="link">Go to Live Dashboard</a>
        </div>
        <script>
        function previewImage(event) {
            const file = event.target.files[0];
            if (!file) return;
            document.getElementById("file-name").innerText = file.name;
            const reader = new FileReader();
            reader.onload = function(){
                const img = document.getElementById('preview');
                img.src = reader.result;
                img.style.display = 'block';
            }
            reader.readAsDataURL(file);
        }
        </script>
    </body>
    </html>"""

# =========================
# DETECT UI (UNCHANGED)
# =========================
@app.post("/detect_ui", response_class=HTMLResponse)
async def detect_ui(file: UploadFile = File(...)):
    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        return HTMLResponse("<h3 style='color:red'>Invalid image uploaded</h3>")

    result = run_detection(frame)
    log_violation(result["violations"])

    _, buffer = cv2.imencode('.jpg', frame)
    img = base64.b64encode(buffer).decode()

    return HTMLResponse(f"""
    <html>
    <body style="background:#020617;color:white;text-align:center">
    <h2>Result</h2>
    <img src="data:image/jpeg;base64,{img}" width="600"/>
    <p>Status: {"VIOLATION" if result["violations"] else "SAFE"}</p>
    <p>{result["violations"]}</p>
    <a href="/ui">Back</a>
    </body>
    </html>
    """)