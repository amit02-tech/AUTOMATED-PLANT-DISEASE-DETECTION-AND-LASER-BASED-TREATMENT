import cv2
import serial
import time
import numpy as np
import math
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# --- SYSTEM GLOBALS ---
system_logs = ["SYSTEM BOOT: AGNI Web Interface Initialized."]


def log_event(message):
    time_str = time.strftime("%H:%M:%S")
    system_logs.append(f"[{time_str}] {message}")
    if len(system_logs) > 50: system_logs.pop(0)  # Keep last 50 logs


# --- CONFIGURATION ---
ARDUINO_PORT = 'COM6'
IP_CAMERA_URL = 'http://192.168.20.3:8080/video'
MODEL_PATH = r"C:\Users\amita\PycharmProjects\PythonProject12\best_mango_disease_model.h5"
CLASSES = ['Anthracnose', 'Background_Noise', 'Bacterial_Canker', 'Cutting_Weevil',
           'Die_Back', 'Gall_Midge', 'Healthy', 'Powdery_Mildew', 'Sooty_Mould']

BAUD_RATE = 9600
COOLDOWN_SECONDS = 3.5
BURN_RADIUS = 50
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# --- LOAD AI & HARDWARE ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    log_event("AI Link: SUCCESS. MobileNetV2 Loaded.")
except Exception as e:
    log_event(f"AI Link: FAILED. {e}")

try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    log_event(f"Hardware Link: SUCCESS on {ARDUINO_PORT}")
except Exception as e:
    log_event(f"Hardware Link: FAILED. {e}")
    arduino = None

cap = cv2.VideoCapture(IP_CAMERA_URL)


# --- THE CORE GENERATOR ---
def generate_frames():
    destroyed_targets = []
    firing_cooldown_end = 0

    while True:
        ret, frame = cap.read()
        if not ret: continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        current_time = time.time()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. FIND LEAF (Update bounds for mango leaves if necessary)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        leaf_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if leaf_contours:
            largest_leaf = max(leaf_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_leaf) > 1000:
                x, y, w, h = cv2.boundingRect(largest_leaf)

                # 2. FEED TO AI
                img_crop = frame[y:y + h, x:x + w]
                img_ai = cv2.resize(img_crop, (224, 224)) / 255.0
                img_ai = np.expand_dims(img_ai, axis=0)

                predictions = model.predict(img_ai, verbose=0)
                class_idx = np.argmax(predictions)
                confidence = predictions[0][class_idx]
                label = CLASSES[class_idx]

                # 3. DISEASE TARGETING LOGIC
                if label not in ['Healthy', 'Background_Noise'] and confidence > 0.75:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} ({confidence * 100:.0f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

                    leaf_roi_hsv = hsv[y:y + h, x:x + w]

                    # Update these via tuner.py for dark Mango spots (Anthracnose/Sooty Mould)
                    lower_disease = np.array([10, 50, 20])
                    upper_disease = np.array([35, 255, 255])

                    disease_mask = cv2.inRange(leaf_roi_hsv, lower_disease, upper_disease)
                    disease_contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if disease_contours:
                        for spot in disease_contours:
                            if cv2.contourArea(spot) > 150:
                                M = cv2.moments(spot)
                                if M["m00"] != 0:
                                    cx_roi = int(M["m10"] / M["m00"])
                                    cy_roi = int(M["m01"] / M["m00"])

                                    target_x = max(0, min(FRAME_WIDTH, x + cx_roi))
                                    target_y = max(0, min(FRAME_HEIGHT, y + cy_roi))

                                    if current_time > firing_cooldown_end:
                                        already_destroyed = any(
                                            math.hypot(target_x - dx, target_y - dy) < BURN_RADIUS for dx, dy in
                                            destroyed_targets)

                                        if not already_destroyed:
                                            if arduino:
                                                arduino.write(f"X{target_x}Y{target_y}\n".encode('utf-8'))

                                            log_event(f"🔥 NEUTRALIZING {label} AT: X:{target_x}, Y:{target_y}")
                                            destroyed_targets.append((target_x, target_y))
                                            firing_cooldown_end = current_time + COOLDOWN_SECONDS
                                        else:
                                            cv2.drawMarker(frame, (target_x, target_y), (0, 0, 0), cv2.MARKER_CROSS, 20,
                                                           2)
                                    else:
                                        cv2.drawMarker(frame, (target_x, target_y), (0, 0, 255), cv2.MARKER_CROSS, 20,
                                                       2)
                else:
                    if confidence > 0.75:
                        color = (0, 255, 0) if label == 'Healthy' else (255, 255, 0)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f"{label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Encode frame for web viewing
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_logs')
def get_logs():
    return jsonify(logs=system_logs)


if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)