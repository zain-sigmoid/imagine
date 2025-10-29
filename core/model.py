from core.themes import THEMES
from types import SimpleNamespace
import os
import io
import base64
from PIL import Image
import requests
import logging
from google import genai
from io import BytesIO
from core.postprocessing import PostProcessing
from google.genai import types

STRENGTH_INDEX = {"Light": 0, "Medium": 1, "Strong": 2}

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


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


class Edit:
    def __init__(self):
        pass

    def edit_with_gemini(self, base_img: Image.Image, prompt: str):
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                return {
                    "status": False,
                    "type": "Key Error",
                    "msg": "GEMINI_API_KEY is not set.",
                }
            else:
                status = False
                client = genai.Client(api_key=api_key)

                # Build request: text + image
                img_bytes = Imagine.pil_to_png_bytes(base_img)
                # parts = [
                #     types.Part.from_bytes(
                #         data=img_bytes,
                #         mime_type="image/png",
                #     ),
                # ]

                # Call Gemini 2.5 Flash Image (preview name may still be required in some environments)
                model_name = (
                    "gemini-2.5-flash-image"  # or "gemini-2.5-flash-image-preview"
                )
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[
                        prompt,
                        types.Part.from_bytes(
                            data=img_bytes, mime_type="image/png"
                        ),  # or use this helper
                    ],
                )

                # Parse response: find first image part
                edited_bytes = None
                if resp and resp.candidates:
                    for part in resp.candidates[0].content.parts:
                        b = Imagine.read_gemini_image_part(part)
                        if b:
                            edited_bytes = b
                            break

                if not edited_bytes:
                    return {"status": False, "msg": "Edit response missing image data."}
                else:
                    edited_img = Image.open(BytesIO(edited_bytes)).convert("RGBA")

                    # Update originals + rebuild your enhanced variants
                    status = True

                    low, med, high = PostProcessing.apply_post_processing(edited_img)
                    result = {
                        "status": status,
                        "type": "success",
                        "images": {
                            "org": edited_img,
                            "low": low,
                            "medium": med,
                            "high": high,
                        },
                    }
                    return result
        except requests.HTTPError as e:
            return {
                "status": False,
                "type": "exception",
                "msg": f"Image edit failed: {e.response.text if e.response is not None else e}",
            }
        except Exception as e:
            return {
                "status": False,
                "type": "exception",
                "msg": f"Unexpected error during edit: {e}",
            }

    def edit_with_openai(base_img: Image.Image, edit_prompt: str):
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return {
                    "status": False,
                    "type": "Key Error",
                    "msg": "OPENAI_API_KEY is not set",
                }
            else:
                size = Imagine.pick_openai_size_from_image(base_img)
                files = {
                    "image": (
                        "image.png",
                        Imagine.pil_to_png_bytes(base_img),
                        "image/png",
                    ),
                }
                data = {
                    "model": "dall-e-2",  # supports edits/inpainting
                    "prompt": edit_prompt,
                    "size": size,
                }
                logger.info("Sending edit request to OpenAI...")
                resp = requests.post(
                    "https://api.openai.com/v1/images/edits",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                    files=files,
                    timeout=90,
                )
                resp.raise_for_status()
                logger.info("Edit response received.")

                payload = resp.json()
                if not payload.get("data"):
                    return {
                        "status": False,
                        "type": "Open AI API Error",
                        "msg": "No image returned from edit.",
                    }
                else:
                    out = payload.get("data", [{}])[0]
                    eimg = None
                    if out.get("b64_json"):
                        # if you forced response_format="b64_json"
                        eimg = Imagine.b64_to_image(out["b64_json"])
                    elif out.get("url"):
                        # fallback: download from signed URL
                        logger.info("Fetching edited image from URL...")
                        r = requests.get(out["url"], timeout=30)
                        r.raise_for_status()
                        eimg = Image.open(io.BytesIO(r.content)).convert("RGBA")
                        # st.image(eimg, caption="Edited Image", width=200)
                    if eimg is None:
                        return {
                            "status": False,
                            "type": "Edit failed",
                            "msg": ("Edit failed: no image returned."),
                        }
                    else:
                        # store + rebuild enhanced variants
                        # st.session_state.images["org"] = img
                        result = {"status": True, "type": "success", "images": eimg}
                        return result

        except requests.HTTPError as e:
            return {
                "status": False,
                "type": "exception",
                "msg": f"Image edit failed: {e.response.text if e.response is not None else e}",
            }
        except Exception as e:
            return {
                "status": False,
                "type": "exception",
                "msg": f"Unexpected error during edit: {e}",
            }
