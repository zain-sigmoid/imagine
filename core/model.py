from core.themes import THEMES
from types import SimpleNamespace
import os
import io
import base64
from PIL import Image
import requests

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

    @staticmethod
    def ensure_mode_rgba(img: Image.Image) -> Image.Image:
        # Convert any mode (RGB, P, CMYK, etc.) to RGBA
        if img.mode != "RGBA":
            return img.convert("RGBA")
        return img

    @staticmethod
    def pil_to_png_bytes(img: Image.Image) -> bytes:
        img = Imagine.ensure_mode_rgba(img)  # <-- ensure acceptable mode
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    @staticmethod
    def pick_openai_size_from_image(img: Image.Image) -> str:
        """Map current image size to a supported square size."""
        w, h = img.size
        m = max(w, h)
        # common choices your app already uses
        candidates = [
            (1024, "1024x1024"),
            (1792, "1792x1024" if w >= h else "1024x1792"),
        ]
        # pick 1024 if unsure
        return (
            "1792x1024"
            if m >= 1400 and w >= h
            else ("1024x1792" if m >= 1400 else "1024x1024")
        )

    @staticmethod
    def read_gemini_image_part(part) -> bytes | None:
        """
        Gemini can return inline_data (bytes) or a URL. Return raw PNG/JPEG bytes.
        """
        if getattr(part, "inline_data", None) and getattr(
            part.inline_data, "data", None
        ):
            return part.inline_data.data  # already bytes
        if getattr(part, "file_data", None) and getattr(part.file_data, "uri", None):
            # Fallback if SDK surfaces uri
            r = requests.get(part.file_data.uri, timeout=60)
            r.raise_for_status()
            return r.content
        if getattr(part, "image_url", None):  # very rare alt surface
            r = requests.get(part.image_url, timeout=60)
            r.raise_for_status()
            return r.content
        return None
