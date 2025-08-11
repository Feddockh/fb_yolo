import os
import requests

HF_TOKEN = os.environ.get("HF_TOKEN")
URL = "https://rp4cmqqmkr9wq302.us-east-1.aws.endpoints.huggingface.cloud"
# https://huggingface.co/hfeddock/yolov8-fireblight

with open("test.png", "rb") as f:
    img_bytes = f.read()

resp = requests.post(
    URL,
    headers={
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "image/png",
    },
    data=img_bytes
)

print("Status:", resp.status_code)
print("Response JSON:", resp.json())