import cv2

# Change to your IP camera URL
IP_CAMERA_URL = 'http://192.168.20.5:8080/video'
cap = cv2.VideoCapture(IP_CAMERA_URL)

print("ALIGNMENT MODE: Center the laser dot on the crosshairs.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (640, 480))

    # Draw the exact center Crosshair (X: 320, Y: 240)
    cv2.line(frame, (320, 0), (320, 480), (0, 255, 0), 2) # Vertical Line
    cv2.line(frame, (0, 240), (640, 240), (0, 255, 0), 2) # Horizontal Line
    cv2.circle(frame, (320, 240), 10, (0, 0, 255), 2)     # Center Target

    cv2.imshow("Boresight Calibration", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()