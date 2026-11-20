import cv2
import serial
import time
import numpy as np
import math
import tensorflow as tf

# --- CONFIGURATION ---
ARDUINO_PORT = 'COM5'  # Change to match your Arduino port
IP_CAMERA_URL = 'http://192.168.20.3:8080/video'  # Replace with your phone's URL, or use 0 for webcam
MODEL_PATH = "best_mango_disease_model.h5"
CLASSES = ['Anthracnose', 'Background_Noise', 'Bacterial_Canker', 'Cutting_Weevil',
           'Die_Back', 'Gall_Midge', 'Healthy', 'Powdery_Mildew', 'Sooty_Mould']

BAUD_RATE = 9600
COOLDOWN_SECONDS = 3.5
BURN_RADIUS = 50
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

print("SYSTEM BOOT: AGNI Autonomous Intervention Initialized.")

# --- LOAD AI BRAIN ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("AI Link: SUCCESS")
except Exception as e:
    print(f"AI Link: FAILED. Cannot load {MODEL_PATH}. Error: {e}")
    exit()

# --- LOAD HARDWARE ---
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("Hardware Link: SUCCESS")
except serial.SerialException as e:
    print(f"Hardware Link: FAILED. Check {ARDUINO_PORT}. Error: {e}")
    arduino = None

cap = cv2.VideoCapture(IP_CAMERA_URL)

# --- SYSTEM MEMORY VARIABLES ---
destroyed_targets = []
firing_cooldown_end = 0

# --- CORE AUTONOMOUS LOOP ---
while True:
    ret, frame = cap.read()
    if not ret: continue

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    current_time = time.time()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. AUTO-DETECT THE LEAF (Green Area Only)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    leaf_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if leaf_contours:
        largest_leaf = max(leaf_contours, key=cv2.contourArea)

        # Lowered to 1000 to detect leaves further away from the camera
        if cv2.contourArea(largest_leaf) > 1000:
            x, y, w, h = cv2.boundingRect(largest_leaf)

            # 2. FEED LEAF TO AI
            img_crop = frame[y:y + h, x:x + w]
            img_ai = cv2.resize(img_crop, (224, 224)) / 255.0
            img_ai = np.expand_dims(img_ai, axis=0)

            predictions = model.predict(img_ai, verbose=0)
            class_idx = np.argmax(predictions)
            confidence = predictions[0][class_idx]
            label = CLASSES[class_idx]

            # 3. IF AI DETECTS DISEASE -> INSTANTLY FIND COORDINATES & FIRE
            if label not in ['Healthy', 'Background_Noise'] and confidence > 0.75:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"THREAT DETECTED: {label} ({confidence * 100:.0f}%)",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Isolate the brown/diseased spot automatically
                leaf_roi_hsv = hsv[y:y + h, x:x + w]

                # --- YOUR TUNER NUMBERS ---
                lower_disease = np.array([10, 50, 20])
                upper_disease = np.array([35, 255, 255])

                disease_mask = cv2.inRange(leaf_roi_hsv, lower_disease, upper_disease)

                # 1. ADD THIS DEBUG WINDOW: It shows you exactly what the laser is targeting (White = Target)
                cv2.imshow("Targeting Laser Vision", disease_mask)

                disease_contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if disease_contours:
                    # 2. CHANGE THIS TO A LOOP: Instead of just picking the biggest object, check all of them
                    for spot in disease_contours:
                        # Only target spots that are big enough to be a real disease (ignores tiny noise pixels)
                        if cv2.contourArea(spot) > 150:
                            M = cv2.moments(spot)
                            if M["m00"] != 0:
                                cx_roi = int(M["m10"] / M["m00"])
                                cy_roi = int(M["m01"] / M["m00"])

                                target_x = max(0, min(FRAME_WIDTH, x + cx_roi))
                                target_y = max(0, min(FRAME_HEIGHT, y + cy_roi))

                                # 4. FIRE LASER (SEEK & DESTROY)
                                if current_time > firing_cooldown_end:
                                    already_destroyed = any(
                                        math.hypot(target_x - dx, target_y - dy) < BURN_RADIUS
                                        for dx, dy in destroyed_targets
                                    )

                                    if not already_destroyed:
                                        if arduino:
                                            arduino.write(f"X{target_x}Y{target_y}\n".encode('utf-8'))
                                        print(f"🔥 NEUTRALIZING {label} AT: X:{target_x}, Y:{target_y}")

                                        destroyed_targets.append((target_x, target_y))
                                        firing_cooldown_end = current_time + COOLDOWN_SECONDS
                                    else:
                                        cv2.drawMarker(frame, (target_x, target_y), (0, 0, 0), cv2.MARKER_CROSS, 20, 2)
                                        cv2.putText(frame, "BURNED", (target_x + 10, target_y - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            else:
                                cv2.putText(frame, "LASER ACTIVE - STANDBY", (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                cv2.drawMarker(frame, (target_x, target_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            else:
                if confidence > 0.75:
                    if label == 'Healthy':
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"HEALTHY ({confidence * 100:.0f}%)", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    elif label == 'Background_Noise':
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                        cv2.putText(frame, f"IGNORED: NOISE ({confidence * 100:.0f}%)", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("AGNI - Autonomous Crop Protection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino: arduino.close()