import requests
import json
import os

# Wczytaj payload z pliku messages_payload.json
payload_path = os.path.join(os.path.dirname(__file__), "messages_payload.json")
with open(payload_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

URL = "http://localhost:8000/messages2"

response = requests.post(URL, json=payload)

print("Status:", response.status_code)
print("Response:", response.json())
