import cv2
import numpy as np
import tensorflow as tf
import serial  # NEW: The Serial library
import time

# --- SETUP SERIAL CONNECTION ---
# CHANGE 'COM4' to your actual Arduino Port!
try:
    arduino = serial.Serial('COM10', 9600, timeout=1)
    time.sleep(2)  # Give Arduino 2 seconds to wake up
    print("Serial Connection Established!")
except:
    print("WARNING: Arduino not found on this port.")
    arduino = None

CAMERA_SOURCE = 1
MODEL_PATH = r"C:\Users\amita\PycharmProjects\PythonProject4\best_plant_disease_model.h5"
print("Loading AI Model...")
model = tf.keras.models.load_model(MODEL_PATH)
CLASSES = ['Citruspot', 'Early_Mild_Spotting', 'Fungal_Infected', 'Healthy',
           'Mild_Edge_Damage', 'Senescent', 'Slightly_Diseased', 'Wrinkled_Leaf']

cap = cv2.VideoCapture(CAMERA_SOURCE)


def find_leaf(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 5000:
            x, y, w, h = cv2.boundingRect(largest_contour)
            return x, y, w, h
    return None


def find_disease_target(leaf_crop):
    hsv = cv2.cvtColor(leaf_crop, cv2.COLOR_BGR2HSV)
    lower_disease = np.array([10, 50, 20])
    upper_disease = np.array([30, 255, 200])
    mask = cv2.inRange(hsv, lower_disease, upper_disease)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy, largest_contour
    return None, None, None


print("🚀 Tracking Active. Looking for biological targets...")

while True:
    ret, frame = cap.read()
    if not ret: break

    leaf_box = find_leaf(frame)

    if leaf_box:
        x, y, w, h = leaf_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        leaf_crop = frame[y:y + h, x:x + w]

        img = cv2.resize(leaf_crop, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img, verbose=0)
        class_idx = np.argmax(predictions)
        confidence = predictions[0][class_idx]
        label = CLASSES[class_idx]

        color = (0, 255, 0)

        if label != 'Healthy' and confidence > 0.80:
            color = (0, 0, 255)
            target_x, target_y, contour = find_disease_target(leaf_crop)

            if target_x is not None:
                global_x = x + target_x
                global_y = y + target_y

                cv2.circle(frame, (global_x, global_y), 15, (0, 0, 255), 2)

                # --- NEW: SEND COMMAND TO ARDUINO ---
                if arduino is not None:
                    # Format: X320Y240
                    command = f"X{global_x}Y{global_y}\n"
                    arduino.write(command.encode())  # Send it down the USB cable!

        cv2.putText(frame, f"AI: {label} ({confidence * 100:.1f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Live Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

if arduino is not None:
    arduino.close()
cap.release()
cv2.destroyAllWindows()