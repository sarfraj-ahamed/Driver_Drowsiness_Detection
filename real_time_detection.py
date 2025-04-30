import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import time
from arduino_control import send_command_to_esp32

# Load trained model
def load_model(model_path):                                 
    if not os.path.exists(model_path):
        print("❌ Error: Model file not found!")
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Load preprocessed dataset
def load_preprocessed_data():
    """Loads preprocessed dataset for EAR threshold calculation."""
    data_path = "processed_data.npy"
    labels_path = "labels.npy"

    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        print("❌ Error: Preprocessed dataset not found! Run data_preprocessing.py first.")
        return None, None

    data = np.load(data_path)
    labels = np.load(labels_path)
    return data, labels

# Calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks):
    """Calculates EAR to detect eye closure."""
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
        B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
        C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
        return (A + B) / (2.0 * C)

    try:
        left_eye = [landmarks[i] for i in [33, 159, 158, 133, 153, 145]]
        right_eye = [landmarks[i] for i in [362, 386, 385, 263, 373, 374]]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        return (left_ear + right_ear) / 2.0  # Average EAR for both eyes
    except IndexError:
        return None

# Preprocess frame for better detection
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)  

# Real-time drowsiness detection
def detect_drowsiness():
    alert_active = False  # Flag to control alert thread
    cap = cv2.VideoCapture(0)
    model_path = "models/drowsiness_model.pkl"

    # Load trained model
    model = load_model(model_path)
    if model is None:
        return

    # Load preprocessed data
    data, labels = load_preprocessed_data()
    if data is None or labels is None:
        return
    
    # Compute average EAR threshold from preprocessed dataset
    EAR_THRESHOLD = np.mean(data[:, -1]) - 0.05  # Adaptive threshold

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.8, min_tracking_confidence=0.8, refine_landmarks=True)

    closed_eye_time = 0  # Track the time of closed eyes

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ear = calculate_ear(face_landmarks.landmark)  

                if ear is None:
                    continue  

                # Compare EAR with preprocessed data threshold
                if ear < EAR_THRESHOLD:
                    if closed_eye_time == 0:  
                        closed_eye_time = time.time()
                    elif time.time() - closed_eye_time > 0.5:  # Trigger alert after 0.5 sec
                        if not alert_active:
                            alert_active = True
                            send_command_to_esp32("close", buzzer=0, vibrate=1, duration=3000)  # Vibration only
                else:
                    closed_eye_time = 0  
                    if alert_active:
                        alert_active = False
                        send_command_to_esp32("open")  # Send "open" to ESP32 to turn off buzzer and vibrator

                # **Draw green dots for face landmarks**
                h, w, _ = frame.shape
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Green dots

                # **Display EAR value on screen**
                cv2.putText(frame, f'EAR: {ear:.2f}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            cv2.putText(frame, f'No face detected! Adjust lighting or camera angle.', (40, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the detection
if __name__ == "__main__":
    detect_drowsiness()
