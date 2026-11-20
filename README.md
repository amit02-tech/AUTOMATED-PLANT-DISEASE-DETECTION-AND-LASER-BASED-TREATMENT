# AGNI — Automated Plant Disease Detection & Laser-Based Treatment System

**AGNI** is an autonomous precision-agriculture prototype that detects mango leaf diseases in real time and physically neutralizes infected spots using a 2-axis servo-mounted laser. By combining a fine-tuned MobileNetV2 deep learning model, multi-stage OpenCV targeting, and an Arduino Nano gimbal controller, AGNI provides a targeted, chemical-free alternative to broad-spectrum pesticides.

---

## Key Features
* **Real-Time Multi-Class Classification:** Uses a custom-trained **MobileNetV2** model to classify 9 categories (`Anthracnose`, `Bacterial_Canker`, `Cutting_Weevil`, `Die_Back`, `Gall_Midge`, `Powdery_Mildew`, `Sooty_Mould`, `Healthy`, and `Background_Noise`).
* **Noise Rejection:** Dedicated `Background_Noise` class filters out non-plant objects, tools, and backgrounds to prevent false positives.
* **Precision Laser Targeting:** Two-stage HSV color masking isolates the leaf and calculates the exact centroid coordinates of each diseased spot.
* **Duplicate-Fire Prevention:** Tracks neutralized coordinates in memory (`BURN_RADIUS`) with a cooldown timer so the laser does not repeatedly burn the same spot.
* **Web Command Center:** Dark-themed Flask dashboard featuring a live MJPEG optical feed and a real-time scrolling system log of detections and target coordinates.

---

## Hardware Architecture
* **Microcontroller:** Arduino Nano
* **Actuation:** 2-Axis Pan/Tilt Servo Gimbal (X-Axis: Pin 9, Y-Axis: Pin 10)
* **Intervention:** Laser Module (Signal/Trigger: Pin 7)
* **Optical Sensor:** Mobile phone camera streaming via IP Webcam
* **Frame:** Overhead PVC gantry providing an unobstructed top-down view of the target area
* **Compute:** PC with CUDA-enabled GPU (tested on NVIDIA RTX 3050)

---

## Tech Stack
* **AI / Computer Vision:** TensorFlow / Keras (MobileNetV2), OpenCV, NumPy
* **Backend & UI:** Flask, HTML5, CSS3 (Glassmorphism), Bootstrap 5, JavaScript
* **Hardware Communication:** PySerial, Arduino C++ (`Servo.h`)

---

## Project Structure
```text
AUTOMATED-PLANT-DISEASE-DETECTION-AND-LASER-BASED-TREATMENT/
├── app.py                     # Main Flask server, AI inference, & serial targeting loop
├── train_model.py             # Two-phase MobileNetV2 training & fine-tuning script
├── tuner.py                   # Interactive HSV slider tool for color mask calibration
├── main_system.py             # Standalone OpenCV execution script (headless/local)
├── templates/
│   └── index.html             # AGNI Web Command Center UI
├── .gitignore                 # Excludes .venv, raw dataset, and large .h5 weights
└── README.md