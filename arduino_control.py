import requests

def send_command_to_esp32(command, buzzer=None, vibrate=None, duration=None):
    base_url = "http://192.168.89.41/"  # Replace with your ESP32's IP

    params = {}
    if buzzer is not None:
        params["buzzer"] = buzzer
    if vibrate is not None:
        params["vibrate"] = vibrate
    if duration is not None:
        params["duration"] = duration

    try:
        response = requests.get(base_url + command, params=params, timeout=3)
        response.raise_for_status()
        print(f"✅ ESP32 responded: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to ESP32: {e}")
