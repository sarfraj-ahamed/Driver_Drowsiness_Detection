
# 🚗 Driver Drowsiness Detection with Vehicle Control (ESP32 + Python + OpenCV )

This project is a **real-time driver monitoring system** that detects drowsiness using facial landmarks and controls a vehicle's motor using an **ESP32**. When the driver closes their eyes for more than a few seconds, the system stops the vehicle and triggers buzzer/vibration alerts.

---

## 🔧 Features

- 👁️ Eye Aspect Ratio (EAR) based drowsiness detection
- 🤖 Vehicle motor control using ESP32
- 🔊 Buzzer and vibration alert on drowsiness
- 📡 Wi-Fi-based communication (Python ⇄ ESP32)
- 🟢 Live face landmark visualization
- 🛑 Emergency stop + alert system

---

## 🛠 Hardware Requirements

| Component           | Quantity | Purpose                                |
|--------------------|----------|----------------------------------------|
| ESP32 Dev Module   | 1        | Main microcontroller with WiFi         |
| L298N Motor Driver | 1        | Controls DC motors                     |
| DC Motors (6–12V)  | 4        | Represents vehicle movement            |
| USB Camera / Laptop Camera | 1 | Face & eye tracking                    |
| Buzzer             | 1        | Audio alert                            |
| Vibrator Motor     | 1        | Haptic alert                           |
| Jumper Wires       | N/A      | Circuit connections                    |
| Power Supply       | 1        | ESP32 + Motor Driver                   |

---

## 💻 Software Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Flask or WebServer (ESP32)
- Arduino IDE (ESP32 setup)

---

## 📁 File Structure

```
DriverDrowsinessDetection/
├── main.py                      # Real-time detection (Python)
├── arduino_control.py          # HTTP-based ESP32 controller
├── models/
│   └── drowsiness_model.pkl    # Trained ML model
├── processed_data.npy          # Preprocessed EAR data
├── labels.npy                  # Associated labels
├── esp_code.ino                # Final ESP32 firmware
├── README.md                   # Documentation
```

---

## 📦 Installation Steps

### 1. Clone this repository

```bash
git clone https://github.com/yourname/driver-drowsiness-esp32.git
cd driver-drowsiness-esp32
```

### 2. Install Python dependencies

```bash
pip install opencv-python mediapipe numpy scikit-learn
```

### 3. Upload ESP32 Code

- Open `esp_code.ino` in Arduino IDE
- Install **ESP32 Board** support via Board Manager
- Select **ESP32 Dev Module**
- Update WiFi SSID and Password
- Upload the code

### 4. Run Python Code

```bash
python main.py
```

---

## ⚙️ How It Works

- Python uses **MediaPipe** to track eye landmarks in real-time.
- EAR (Eye Aspect Ratio) is computed to detect if eyes are closed.
- If eyes remain closed for >0.5 sec, Python sends a command to ESP32 via WiFi (`/close`).
- ESP32 activates **buzzer + vibrator** and **stops motors**.
- If the driver's eyes open again, the system sends `/open` to resume movement.

---

## 📈 EAR Calculation

EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)

Where p1–p6 are eye contour points from MediaPipe.

---

## 🚀 Future Enhancements

- Integrate **GPS + GSM** for SOS messaging
- Use **IR camera** for night driving
- Add **face recognition** for driver authentication
- Integrate with **vehicle OBD port** for real-car control

---

## 👨‍💻 Authors

- Your Name – [Sarfraj Ahamed](https://github.com/sarfraj-ahamed)
- Collaborator – [Sivarama Krishnan](https://github.com/MSRAM-NEC) , [Ram Praveen](https://github.com/praveen-asha) , [Ashwin Krishna](https://github.com/Ashwin-456)

---

## 📄 License

This project is licensed under the MIT License.
