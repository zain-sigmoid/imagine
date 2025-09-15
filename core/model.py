from core.themes import THEMES
from types import SimpleNamespace
import os
import io
import base64
from PIL import Image

STRENGTH_INDEX = {"Light": 0, "Medium": 1, "Strong": 2}


class Imagine:
    @staticmethod
    def build_final_prompt(subject: str, theme_name: str, strength_label: str) -> str:
        subject = (subject or "").strip()
        style = THEMES.get(theme_name, ["", "", ""])[STRENGTH_INDEX[strength_label]]
        if style:
            return (
                f"{subject}\n\n"
                f"Style: {style}. "
                f"Avoid text, watermarks, signatures, borders."
            )
        return subject

    @staticmethod
    def mock_response_from_file(path: str):
        # Read and encode the image
        with open(path, "rb") as f:
            img_bytes = f.read()
        b64_str = base64.b64encode(img_bytes).decode("utf-8")

        # Match OpenAI's response format: resp.data[0].b64_json
        fake_item = SimpleNamespace(b64_json=b64_str, url=None)
        fake_resp = SimpleNamespace(data=[fake_item])
        return fake_resp

    @staticmethod
    def b64_to_image(b64_str: str) -> Image.Image:
        img_bytes = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_bytes)).convert("RGBA")
