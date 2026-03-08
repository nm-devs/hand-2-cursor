import cv2
import time
from core.sign_classifier import SignClassifier
from core.hand_detector import HandDetector
from utils.text_overlay import draw_prediction
from core.feature_extractor import FeatureExtractor
from utils.prediction_smoother import PredictionSmoother
from controllers.mouse_controller import MouseController
from controllers.sign_language_controller import SignLanguageController
from config import (
    CAMERA_INDEX, CAM_WIDTH, CAM_HEIGHT,
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
    SMOOTHING_ALPHA, WINDOW_TITLE,
    SMOOTHING_WINDOW_SIZE, SMOOTHING_DOMINANCE_THRESHOLD
)

# Initialize controllers
controllers = {
    "mouse": MouseController(alpha=SMOOTHING_ALPHA),
    "sign_language": SignLanguageController(),
}

# Initialize hand detector (start in single hand mode)
detector = HandDetector(1, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE)

fe = FeatureExtractor(use_z=False)  # Must match training config

# Load trained model
try:
    classifier = SignClassifier('models/trained_model.pkl')
except FileNotFoundError:
    print("Trained model not found. Please run 'train_model.py' first to create the model file.")
    classifier = None

# Initialize webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# Check if camera opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Runtime state
prev_time = 0
mode = "mouse"

# hand detection mode toggle
max_hands_mode = 1 #start with single hand mode

# Initialize prediction smoother
smoother = PredictionSmoother(window_size=SMOOTHING_WINDOW_SIZE, dominance_threshold=SMOOTHING_DOMINANCE_THRESHOLD)
displayed_sign = None
displayed_confidence = None

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

        # Extract and normalize features
        features = fe.extract(landmarks)
        normalized_features = fe.normalize(features)

        # Predict gesture
        if classifier is not None:
            label, confidence = classifier.predict(normalized_features)
            smoother.add_prediction(label)
            stable = smoother.get_stable()

            # Update displayed sign if stable prediction is available
            if stable is not None:
                displayed_sign = stable
                displayed_confidence = confidence

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # Display FPS and mode
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Hands: {max_hands_mode}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Mode: {mode}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display sign if available
    if displayed_sign and displayed_confidence:
        draw_prediction(frame, displayed_sign, displayed_confidence)

    cv2.imshow(WINDOW_TITLE, frame)

    # Handle keypresses
    """
    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        mode = "sign_language" if mode == "mouse" else "mouse"
    if key == 27 or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
        break
    """
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('m'):
        mode = "sign_language" if mode == "mouse" else "mouse"
    """
    # making the code raedy for future changes regarding translating of 2 hands signing
    if key == ord('1'):
        max_hands_mode = 1
        detector.hands.max_num_hands = 1
        print('Hand detection mode: 1 hand')
    if key == ord('2'):
        max_hands_mode = 2
        detector.hands.max_num_hands = 2
        print('Hand detection mode: 2 hands')
    """
    if key == 27 or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()