# 🤟 SignSense

**Real-time Sign Language Interpreter using Computer Vision**

A Python-based hand tracking and gesture recognition system that uses MediaPipe and OpenCV to detect hand landmarks, identify individual finger positions, and interpret sign language in real-time.

---

## ✨ Features

- 🖐️ **Real-time Hand Detection** — Tracks up to 2 hands simultaneously
- 🎯 **21-Point Landmark Tracking** — Precise finger joint detection
- 🔄 **Live Webcam Feed** — Mirror-mode display with FPS counter
- 🏷️ **Hand Classification** — Distinguishes between left and right hands
- 📦 **Modular Architecture** — Clean separation of detection and drawing utilities

## 🛠️ Tech Stack

- **Python 3.9–3.11** (recommended for MediaPipe compatibility)
- **OpenCV** — Video capture and image processing
- **MediaPipe** — Hand landmark detection

## 📁 Project Structure

```
hand-2-cursor/
├── main.py                         # Entry point (mode switching, webcam loop)
├── config.py                       # Project-wide constants
├── requirements.txt                # Python dependencies
├── README.md
├── ROADMAP.md                      # Development roadmap & reference
├── .gitignore
│
├── core/                           # Detection, classification, NLP
│   ├── hand_detector.py            # Hand landmark detection (MediaPipe)
│   ├── feature_extractor.py        # Landmark → feature vector
│   ├── sign_classifier.py          # Sign language classifier
│   └── sentence_builder.py         # Word/sentence accumulation
│
├── controllers/                    # Mode controllers
│   ├── mouse_controller.py         # Gesture-based mouse control
│   └── sign_language_controller.py # Sign language interpreter mode
│
├── utils/                          # Drawing & display helpers
│   ├── drawing_utils.py            # Hand landmarks, skeleton, bounding box
│   └── text_overlay.py             # On-screen text display
│
├── data/                           # Data collection & preprocessing
│   ├── collect_images.py           # Webcam image capture script
│   └── extract_landmarks.py        # Landmark extraction script
│
├── models/                         # Training scripts & saved weights
│   └── train_model.py              # Model training script
│
└── assets/                         # Static assets & reference images
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python mediapipe
```

### Run

```bash
python main.py
```

Press **ESC** or click the **X** button to exit.

## 🗺️ Roadmap

- [ ] Finger state detection (open/closed)
- [ ] Finger counting
- [ ] Basic sign language gesture recognition
- [ ] ASL alphabet interpretation

---

Made with ❤️ and Python
Done By Michael Musallam and Nadim Baboun