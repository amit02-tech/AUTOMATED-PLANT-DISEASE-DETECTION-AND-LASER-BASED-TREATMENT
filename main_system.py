import cv2
import serial
import time
import numpy as np
import math

print("SYSTEM BOOT: S.O.L.A.R.A Initialized.")

# !!! CHANGE THESE !!!
ARDUINO_PORT = ('COM8')
IP_CAMERA_URL = 'http://192.168.20.5:8080/video'

try:
    arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
    time.sleep(2)
    print("Hardware Link: SUCCESS")
except Exception as e:
    print(f"Hardware Link: FAILED. Error: {e}")
    exit()

cap = cv2.VideoCapture(IP_CAMERA_URL)

# --- NEW: MEMORY AND COOLDOWN VARIABLES ---
destroyed_targets = []  # Memory list of burned spots
firing_cooldown_end = 0  # Timer to pause the AI while Arduino shoots

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (640, 480))
    current_time = time.time()

    # 1. LEAF DETECTION (Green)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    leaf_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if leaf_contours:
        largest_leaf = max(leaf_contours, key=cv2.contourArea)

        if cv2.contourArea(largest_leaf) > 3000:
            x, y, w, h = cv2.boundingRect(largest_leaf)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 2. DISEASE DETECTION (Brown inside the leaf)
            leaf_roi_hsv = hsv[y:y + h, x:x + w]
            lower_brown = np.array([10, 50, 20])
            upper_brown = np.array([30, 255, 200])
            brown_mask = cv2.inRange(leaf_roi_hsv, lower_brown, upper_brown)
            disease_contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if disease_contours:
                largest_disease = max(disease_contours, key=cv2.contourArea)

                if cv2.contourArea(largest_disease) > 200:
                    M = cv2.moments(largest_disease)
                    if M["m00"] != 0:
                        cx_roi = int(M["m10"] / M["m00"])
                        cy_roi = int(M["m01"] / M["m00"])

                        target_x = x + cx_roi
                        target_y = y + cy_roi

                        # --- NEW: THE SEEK AND DESTROY LOGIC ---
                        if current_time > firing_cooldown_end:
                            # Check if we already destroyed this spot
                            already_destroyed = False
                            for (dx, dy) in destroyed_targets:
                                # If the target is within 50 pixels of an old burn mark, ignore it
                                distance = math.sqrt((target_x - dx) ** 2 + (target_y - dy) ** 2)
                                if distance < 50:
                                    already_destroyed = True
                                    break

                            if not already_destroyed:
                                # IT IS A NEW TARGET! Send to Arduino.
                                command = f"X{target_x}Y{target_y}\n"
                                arduino.write(command.encode('utf-8'))
                                print(f"FIRING AT: {target_x}, {target_y}")

                                # Add to memory and start a 3.5 second cooldown
                                destroyed_targets.append((target_x, target_y))
                                firing_cooldown_end = current_time + 3.5
                            else:
                                # We already burned this one, mark it black on the screen
                                cv2.drawMarker(frame, (target_x, target_y), (0, 0, 0), cv2.MARKER_CROSS, 20, 2)
                                cv2.putText(frame, "BURNED", (target_x + 10, target_y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 0), 2)

                        else:
                            # We are currently waiting for the Arduino to finish its laser sequence
                            cv2.putText(frame, "FIRING LASER - STANDBY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 3)
                            cv2.drawMarker(frame, (target_x, target_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    cv2.imshow("Cacao Guard - Seek & Destroy", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()