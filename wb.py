from flask import Flask, render_template, Response, jsonify
import cv2
import serial
import time
from datetime import datetime
import numpy as np
import os
import winsound
import pyttsx3
import threading
import tensorflow as tf

# --- FLASK SETUP ---
template_dir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(template_dir, 'templates')
app = Flask(__name__, template_folder=template_dir)

# --- SYSTEM CONFIGURATION ---
IP_CAMERA_URL = 'http://100.88.137.95:8080/video'
ARDUINO_PORT = 'COM10'
MODEL_PATH = "best_plant_disease_model.h5"

# --- AI MODEL LOADING ---
print("AI BOOT: Loading MobileNetV2 Brain...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    CLASSES = ['Citruspot', 'Early_Mild_Spotting', 'Fungal_Infected', 'Healthy',
               'Mild_Edge_Damage', 'Senescent', 'Slightly_Diseased', 'Wrinkled_Leaf']
    print("AI BOOT: Model Loaded Successfully.")
except Exception as e:
    print(f"AI BOOT: CRITICAL ERROR - Could not load model: {e}")
    model = None

# --- GLOBAL VARIABLES ---
SYSTEM_ACTIVE = True
last_tracking_time = 0  # Replaces the "Destroyed Targets" memory
system_logs = []


# --- AUDIO & LOGGING ENGINES ---
def add_log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_string = f"[{timestamp}] {message}"
    system_logs.insert(0, log_string)
    if len(system_logs) > 50: system_logs.pop()
    print(log_string)


def play_alarm(message):
    def alarm_thread():
        winsound.Beep(2500, 300)
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        engine.say(message)
        engine.runAndWait()

    threading.Thread(target=alarm_thread, daemon=True).start()


# --- TARGETING LOGIC ---
def find_disease_target(leaf_roi_hsv, label):
    if label in ['Fungal_Infected', 'Slightly_Diseased', 'Early_Mild_Spotting']:
        lower_disease = np.array([15, 40, 40])
        upper_disease = np.array([35, 255, 255])
    else:
        lower_disease = np.array([10, 50, 20])
        upper_disease = np.array([30, 255, 200])

    mask = cv2.inRange(leaf_roi_hsv, lower_disease, upper_disease)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
    return None, None


# --- HARDWARE BOOT SEQUENCE ---
add_log("SYSTEM BOOT: S.O.L.A.R.A Tracking Mode Online.")
play_alarm("S.O.L.A.R.A. systems online.")

try:
    arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
    time.sleep(2)
    add_log("HARDWARE: Arduino connected on " + ARDUINO_PORT)
except Exception as e:
    add_log(f"WARNING: Arduino not found on {ARDUINO_PORT}.")
    arduino = None


# --- CORE VISION & TARGETING LOOP ---
def generate_frames():
    global SYSTEM_ACTIVE, last_tracking_time
    cap = cv2.VideoCapture(IP_CAMERA_URL)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        frame = cv2.resize(frame, (640, 480))
        current_time = time.time()

        if SYSTEM_ACTIVE and model is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 1. LEAF DETECTION
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            leaf_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if leaf_contours:
                largest_leaf = max(leaf_contours, key=cv2.contourArea)
                if cv2.contourArea(largest_leaf) > 1500:
                    x, y, w, h = cv2.boundingRect(largest_leaf)
                    leaf_roi_hsv = hsv[y:y + h, x:x + w]

                    # 2. AI DIAGNOSIS
                    img_crop = frame[y:y + h, x:x + w]
                    img_ai = cv2.resize(img_crop, (224, 224))
                    img_ai = img_ai / 255.0
                    img_ai = np.expand_dims(img_ai, axis=0)

                    predictions = model.predict(img_ai, verbose=0)
                    class_idx = np.argmax(predictions)
                    confidence = predictions[0][class_idx]
                    label = CLASSES[class_idx]

                    # 3. CONTINUOUS TRACKING LOGIC
                    color = (0, 255, 0)  # Green for Healthy
                    if label != 'Healthy' and confidence > 0.60:
                        color = (0, 0, 255)  # Red for Disease
                        cx_roi, cy_roi = find_disease_target(leaf_roi_hsv, label)

                        if cx_roi is not None:
                            target_x = x + cx_roi
                            target_y = y + cy_roi

                            # Update the Arduino every 0.5 seconds
                            if current_time - last_tracking_time > 0.5:
                                if arduino:
                                    arduino.write(f"X{target_x}Y{target_y}\n".encode('utf-8'))
                                add_log(f"TRACKING: {label} at X:{target_x} Y:{target_y}")
                                last_tracking_time = current_time

                            cv2.drawMarker(frame, (target_x, target_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

                    # UI Elements
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} {confidence * 100:.0f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# --- FLASK ROUTES ---
@app.route('/')
def index(): return render_template('index.html')


@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_logs')
def get_logs(): return jsonify(logs=system_logs)


@app.route('/toggle_override', methods=['POST'])
def toggle_override():
    global SYSTEM_ACTIVE
    SYSTEM_ACTIVE = not SYSTEM_ACTIVE
    return jsonify(system_active=SYSTEM_ACTIVE)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)