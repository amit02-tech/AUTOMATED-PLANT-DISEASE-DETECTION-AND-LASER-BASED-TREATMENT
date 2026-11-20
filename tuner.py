import cv2
import numpy as np

# --- CONFIGURATION ---
# Change to your Wi-Fi URL if using the IP camera, or leave as 0 for webcam
CAMERA_SOURCE = 0


def main():
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    # Allow the camera to warm up and adjust its auto-exposure
    for _ in range(10):
        ret, frame = cap.read()

    if not ret:
        print("Error: Could not grab a frame from the camera.")
        return

    frame = cv2.resize(frame, (640, 480))

    print("INSTRUCTIONS:")
    print("1. A window will open with your camera feed.")
    print("2. Click and drag a box strictly around the brown/infected fungal spot.")
    print("3. Press ENTER or SPACE to confirm the selection.")

    # Open the ROI selector GUI
    roi = cv2.selectROI("AGNI Auto-Tuner", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("AGNI Auto-Tuner")

    x, y, w, h = roi

    if w > 0 and h > 0:
        # Crop to just the disease spot you selected
        disease_crop = frame[y:y + h, x:x + w]

        # Convert crop to HSV
        hsv_crop = cv2.cvtColor(disease_crop, cv2.COLOR_BGR2HSV)

        # Find the absolute minimum and maximum HSV values inside your drawn box
        h_min, s_min, v_min = np.min(hsv_crop, axis=(0, 1))
        h_max, s_max, v_max = np.max(hsv_crop, axis=(0, 1))

        # Add a slight padding (tolerance buffer) to make tracking more reliable
        buffer = 15
        lower_bound = np.array([max(0, h_min - buffer), max(0, s_min - buffer), max(0, v_min - buffer)])
        upper_bound = np.array([min(179, h_max + buffer), min(255, s_max + buffer), min(255, v_max + buffer)])

        print("\n✅ AUTO-CALIBRATION COMPLETE. COPY THESE ARRAYS:")
        print("-" * 50)
        print(f"lower_brown = np.array([{lower_bound[0]}, {lower_bound[1]}, {lower_bound[2]}])")
        print(f"upper_brown = np.array([{upper_bound[0]}, {upper_bound[1]}, {upper_bound[2]}])")
        print("-" * 50)
    else:
        print("\n❌ Calibration aborted. No box was drawn.")

    cap.release()


if __name__ == "__main__":
    main()