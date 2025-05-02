import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import time
import geocoder
from telegram import Bot
from arduino_control import send_command_to_esp32
import threading
import asyncio


# Configuration
TELEGRAM_BOT_TOKEN = #give the telegram bot id
TELEGRAM_CHAT_ID =  # Your group chat ID
ESP32_ALERT_DURATION = 0.5  # Seconds of closed eyes to trigger ESP32
SOS_TRIGGER_TIME = 5        # 5 Seconds (90 seconds) for SOS

# Load trained model
def load_model(model_path):                                
    if not os.path.exists(model_path):
        print("❌ Error: Model file not found!")
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Load preprocessed dataset
def load_preprocessed_data():
    data_path = "processed_data.npy"
    labels_path = "labels.npy"
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        print("❌ Error: Preprocessed dataset not found! Run data_preprocessing.py first.")
        return None, None
    return np.load(data_path), np.load(labels_path)

# Calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks):
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
        B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
        C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
        return (A + B) / (2.0 * C)

    try:
        left_eye = [landmarks[i] for i in [33, 159, 158, 133, 153, 145]]
        right_eye = [landmarks[i] for i in [362, 386, 385, 263, 373, 374]]
        return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    except IndexError:
        return None

# Preprocess frame for better detection
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

# Telegram SOS Alert
def send_sos_alert():
    try:
        location = geocoder.ip('me')
        if location.ok:
            maps_url = f"https://www.google.com/maps?q={location.latlng[0]},{location.latlng[1]}"
            message = (
                "🚨 **EMERGENCY: DRIVER UNCONSCIOUS** 🚨\n\n"
                f"📍 **Location:** {location.city}, {location.country}\n"
                f"🌐 **Google Maps:** [Click here]({maps_url})\n\n"
                "⚠️ Driver has been unresponsive for 1.5+ minutes!\n"
                "Alert sent to family, medical team, and authorities."
            )
            async def send_telegram_alert():
                bot = Bot(token=TELEGRAM_BOT_TOKEN)
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🚨 SOS alert: Drowsiness detected!")
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID,text=message)

            # Call the coroutine
            asyncio.run(send_telegram_alert())
            print("SOS alert sent to Telegram!")
    except Exception as e:
        print(f"Failed to send SOS: {e}")

# Main Detection Function
def detect_drowsiness():
    cap = cv2.VideoCapture(0)
    model = load_model("models/drowsiness_model.pkl")
    data, labels = load_preprocessed_data()
    if model is None or data is None:
        return

    EAR_THRESHOLD = np.mean(data[:, -1]) - 0.05
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        refine_landmarks=True
    )

    # State trackers
    closed_eye_time = 0
    no_face_time = None
    alert_active = False
    sos_sent = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            no_face_time = None  # Reset no-face timer
            for face_landmarks in results.multi_face_landmarks:
                ear = calculate_ear(face_landmarks.landmark)
                if ear is None:
                    continue

                # ESP32 Alert Logic
                if ear < EAR_THRESHOLD:
                    if closed_eye_time == 0:
                        closed_eye_time = time.time()
                    elif time.time() - closed_eye_time > ESP32_ALERT_DURATION and not alert_active:
                        alert_active = True
                        send_command_to_esp32("close", buzzer=0, vibrate=1, duration=3000)  # Vibration only
                       
                    # SOS Condition (1.5 minutes of closed eyes)
                    if time.time() - closed_eye_time > SOS_TRIGGER_TIME and not sos_sent:
                        threading.Thread(target=send_sos_alert).start()
                        sos_sent = True
                else:
                    closed_eye_time = 0
                    if alert_active:
                        alert_active = False 
                        send_command_to_esp32("open")
                    sos_sent = False

                # Draw face landmarks
                h, w, _ = frame.shape
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Display EAR value
                cv2.putText(frame, f'EYE: {ear:.2f}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            # No face detected warning
            cv2.putText(frame, "No face detected! Adjust lighting or camera angle.", (40, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           
            # Start no-face timer
            if no_face_time is None:
                no_face_time = time.time()
            elif time.time() - no_face_time > SOS_TRIGGER_TIME and not sos_sent:
                threading.Thread(target=send_sos_alert).start()
                sos_sent = True

        # Show countdown timer if eyes are closed
        if closed_eye_time > 0:
            elapsed = int(time.time() - closed_eye_time)
            remaining = max(0, SOS_TRIGGER_TIME - elapsed)
            cv2.putText(frame, f"SOS in: {remaining}s", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_drowsiness()
