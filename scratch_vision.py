import os
import fitz
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Create a dummy image
import numpy as np
from PIL import Image, ImageDraw, ImageFont

img = Image.new('RGB', (800, 600), color = (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((10,10), "Jane Doe\nSoftware Engineer\nSkills: Python, Java, C++\nExperience: 5 years at Google", fill=(0,0,0))
img.save('test_resume.jpg')

# Encode image
with open('test_resume.jpg', 'rb') as f:
    img_bytes = f.read()

encoded_string = base64.b64encode(img_bytes).decode('utf-8')

response = client.chat.completions.create(
    model="llama-3.2-90b-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all the text from this resume image exactly as it appears. Do not add any extra commentary or markdown formatting, just return the raw text."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
            ]
        }
    ],
    temperature=0.1
)
print("VISION OUTPUT:", response.choices[0].message.content)
