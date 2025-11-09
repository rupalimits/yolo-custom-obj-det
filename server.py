from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from ultralytics import YOLO
import cv2, csv, os, time, threading, json
import easyocr
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"


# ------------------ Flask setup ------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    static_folder=STATIC_FOLDER,     # serve static files from ./static
    static_url_path="/static"        # access via /static/...
)

# ------------------ Shared state ------------------
progress = {
    "frame": 0,
    "total": 0,
    "done": False,
    "csv_path": None,
    "percent": 0,
    "new_data": [],
    "plates": []
}
lock = threading.Lock()

# ------------------ Serve index.html ------------------
@app.route("/")
def serve_index():
    """Serve index.html from current directory"""
    return send_from_directory(BASE_DIR, "index.html")

# ------------------ Upload Route ------------------
@app.route("/upload", methods=["POST"])
def upload_video():
    global progress
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video uploaded"}), 400

    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    # Reset progress
    progress = {"frame": 0, "total": 0, "done": False,
                "csv_path": None, "percent": 0, "new_data": [],
                "plates": []}

    # Start YOLO processing in background
    threading.Thread(target=process_video, args=(filename,), daemon=True).start()

    return jsonify({"status": "processing started", "file": file.filename})

# ------------------ OCR Setup ------------------
ENABLE_OCR = True
reader = easyocr.Reader(['en'], gpu=False)

dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}


def check_license_plate_format(text):
    if len(text) != 7:
        return False
    valid = (
        (text[0].isupper() or text[0] in dict_int_to_char)
        and (text[1].isupper() or text[1] in dict_int_to_char)
        and (text[2].isdigit() or text[2] in dict_char_to_int)
        and (text[3].isdigit() or text[3] in dict_char_to_int)
        and (text[4].isupper() or text[4] in dict_int_to_char)
        and (text[5].isupper() or text[5] in dict_int_to_char)
        and (text[6].isupper() or text[6] in dict_int_to_char)
    )
    return valid


def format_license_number(text):
    license_number = ''
    char_map = {
        0: dict_int_to_char,
        1: dict_int_to_char,
        4: dict_int_to_char,
        5: dict_int_to_char,
        6: dict_int_to_char,
        2: dict_char_to_int,
        3: dict_char_to_int,
    }
    for i in range(7):
        if text[i] in char_map[i]:
            license_number += char_map[i][text[i]]
        else:
            license_number += text[i]
    return license_number


def read_license_plate(img):
    # print("Hello")
    detection = reader.readtext(img)
    for det in detection:
        _, text, score = det
        text = text.upper().replace(' ', '')
        if check_license_plate_format(text):
            # print("Return")
            return format_license_number(text), score
    return None, None


def map_plate_to_car(plate_box, car_boxes):
    x1p, y1p, x2p, y2p, score, class_id = plate_box
    for car_box in car_boxes:
        if car_box.id is None:
            continue
        x1c, y1c, x2c, y2c = [float(v) for v in car_box.xyxy[0]]
        if x1p > x1c and y1p > y1c and x2p < x2c and y2p < y2c:
            return int(car_box.id.item()), (x1c, y1c, x2c, y2c)
    return None, None





def preprocess_for_ocr(crop, box_size=21):
    """
    Takes an RGB crop and returns a cleaned image for OCR.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    
    # Remove noise using Non-Local Means Denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        box_size, 2
    )
    
    # Morphological operations to remove small noise and connect text
    kernel = np.ones((2,2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Close small gaps
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)     # Remove small dots
    
    # Resize to improve OCR on small text
    scale = 2
    clean = cv2.resize(clean, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    return clean


# ------------------ Video Processing ------------------
def process_video(video_path):
    """Run YOLO detection frame by frame and stream progress"""
    global progress

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = f"uploads/detections-{video_name}.csv"
    progress['csv_path'] = csv_path

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress["total"] = total_frames

    model = YOLO("yolo11n.pt")  # main object tracker
    plate_model = YOLO("license_plate_detector.pt")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "id", "class", "confidence", "x1", "y1", "x2", "y2", "parent", "value"])
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", stream=False, verbose=False)
            detections = []

            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes.id is not None:
                    xyxys = boxes.xyxy.cpu().numpy()
                    ids = boxes.id.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy()

                    for i, track_id in enumerate(ids):
                        x1, y1, x2, y2 = map(float, xyxys[i])
                        cls_idx = int(clss[i])
                        conf = float(confs[i])
                        cls_name = model.names[cls_idx]
                        detections.append({
                            "frame": int(frame_id),
                            "id": int(track_id),
                            "class": cls_name,
                            "confidence": conf,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "parent": 0,
                            "value": ""
                        })
                        writer.writerow([frame_id, int(track_id), cls_name, conf, x1, y1, x2, y2, 0, ''])

            # License plate detection
            plate_results = plate_model(frame, verbose=False)[0]
            plates = []
            plate_numbers = []

            if ENABLE_OCR:
                for plate in plate_results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = plate                
                    car_id, _ = map_plate_to_car(plate, boxes)
    
                    plates.append({
                        "frame": frame_id, 
                        "id": car_id, 
                        "class": "number_plate", 
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2, 
                        "confidence": score,
                        "parent": car_id,
                        "value": ""
                    })
                    writer.writerow([frame_id, car_id, "number_plate", score, x1, y1, x2, y2, 0, ''])
                    
                    
                    cropped_plate = frame[int(y1):int(y2), int(x1):int(x2)]
                    if cropped_plate.size == 0:
                        continue

                        
                    cropped_plate_gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                    _, plate_thresholded = cv2.threshold(cropped_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
                    # plate_thresholded = preprocess_for_ocr(cropped_plate)

                    
                    license_number, score = read_license_plate(plate_thresholded)
                    # license_number = pytesseract.image_to_string(plate_thresholded)
                    if license_number:
                        
                        if car_id is not None:
                            # print(car_id, license_number, score)
                            plate_numbers.append({
                                "frame": frame_id, 
                                "id": 0, 
                                "class": "text", 
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2, 
                                "parent": car_id,
                                "confidence": score,
                                "value": license_number
                            })
                            writer.writerow([frame_id, car_id, "text", score, x1, y1, x2, y2, car_id, license_number])

            f.flush()

            with lock:
                progress.update({
                    "frame": frame_id,
                    "percent": round((frame_id / total_frames) * 100, 2),
                    "new_data": (detections + plates) + plate_numbers,
                })

    cap.release()
    with lock:
        progress.update({"csv_path": csv_path, "done": True, "percent": 100})

# ------------------ SSE ------------------
@app.route("/progress")
def progress_stream():
    def generate():
        last_sent = 0
        while True:
            with lock:
                if progress["frame"] != last_sent:
                    data = {
                        "frame": progress["frame"],
                        "total": progress["total"],
                        "percent": progress["percent"],
                        "done": progress["done"],
                        "new_data": progress["new_data"],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    last_sent = progress["frame"]

                if progress["done"]:
                    yield f"data: {json.dumps(progress)}\n\n"
                    break
            time.sleep(0.2)

    return Response(generate(), mimetype="text/event-stream")

# ------------------ CSV Download ------------------
# @app.route("/csv")
# def get_csv():
#     if progress["csv_path"] and os.path.exists(progress["csv_path"]):
#         return send_file(progress["csv_path"], as_attachment=True)
#     return jsonify({"error": "CSV not ready"}), 404


@app.route('/uploads/<path:filename>')
def serve_files(filename):
    return send_from_directory('uploads', filename)
# ------------------ Run ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)
