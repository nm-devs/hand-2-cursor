import cv2
import time
from core.hand_detector import HandDetector
from controllers.mouse_controller import MouseController
from controllers.sign_language_controller import SignLanguageController
from core.sign_classifier import SignClassifier
from core.feature_extractor import FeatureExtractor
from config import (
    CAMERA_INDEX, CAM_WIDTH, CAM_HEIGHT,
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
    SMOOTHING_ALPHA, WINDOW_TITLE,
)

# Initialize controllers
controllers = {
    "mouse": MouseController(alpha=SMOOTHING_ALPHA),
    "sign_language": SignLanguageController(),
}

# Initialize hand detector
detector = HandDetector(MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE)

# Initialize webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
fe = FeatureExtractor(use_z=False)  # Must match training config
classifier = SignClassifier('models/trained_model.pkl')
if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Runtime state
prev_time = 0
mode = "mouse"

# Main loop
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        break

    frame = cv2.flip(frame, 1)
    hands_data = detector.detect(frame)

    # Process first detected hand with the active controller
    if hands_data:
        controllers[mode].process_frame(frame, hands_data[0], detector)
        hand = hands_data[0]
        landmarks = hand['landmarks']

        #extract and normalize features
        features = fe.extract(landmarks)
        normalized_features = fe.normalize(features)

        # predict gesture
        label, confidence = classifier.predict(normalized_features)
        # display result
        cv2.putText(frame, f'{label} ({confidence:.1%})',
                    (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # Display FPS and mode
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Mode: {mode}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(WINDOW_TITLE, frame)

    # Handle keypresses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        mode = "sign_language" if mode == "mouse" else "mouse"
    if key == 27 or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()