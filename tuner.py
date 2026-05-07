import cv2
import numpy as np

CAMERA_SOURCE = 0 # Change to your Wi-Fi URL if needed

def nothing(x):
    pass

# Open Camera
cap = cv2.VideoCapture(CAMERA_SOURCE)

# Create a window with sliders (Trackbars)
cv2.namedWindow("HSV Tuner")
cv2.createTrackbar("Hue Min", "HSV Tuner", 0, 179, nothing)
cv2.createTrackbar("Hue Max", "HSV Tuner", 179, 179, nothing)
cv2.createTrackbar("Sat Min", "HSV Tuner", 0, 255, nothing)
cv2.createTrackbar("Sat Max", "HSV Tuner", 255, 255, nothing)
cv2.createTrackbar("Val Min", "HSV Tuner", 0, 255, nothing)
cv2.createTrackbar("Val Max", "HSV Tuner", 255, 255, nothing)

# Default starting positions (Roughly Brown/Yellow)
cv2.setTrackbarPos("Hue Min", "HSV Tuner", 5)
cv2.setTrackbarPos("Hue Max", "HSV Tuner", 35)
cv2.setTrackbarPos("Sat Min", "HSV Tuner", 50)
cv2.setTrackbarPos("Val Min", "HSV Tuner", 20)

print("Slide the bars until ONLY the disease is white, and the background is black.")

while True:
    ret, frame = cap.read()
    if not ret: break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read the current slider positions
    h_min = cv2.getTrackbarPos("Hue Min", "HSV Tuner")
    h_max = cv2.getTrackbarPos("Hue Max", "HSV Tuner")
    s_min = cv2.getTrackbarPos("Sat Min", "HSV Tuner")
    s_max = cv2.getTrackbarPos("Sat Max", "HSV Tuner")
    v_min = cv2.getTrackbarPos("Val Min", "HSV Tuner")
    v_max = cv2.getTrackbarPos("Val Max", "HSV Tuner")

    # Apply the mask
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # Show the original and the mask
    cv2.imshow("Original", frame)
    cv2.imshow("Mask (Aim for Disease = White)", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\n✅ YOUR PERFECT NUMBERS ARE:")
        print(f"lower_disease = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"upper_disease = np.array([{h_max}, {s_max}, {v_max}])")
        break

cap.release()
cv2.destroyAllWindows()