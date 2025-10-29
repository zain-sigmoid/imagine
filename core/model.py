from core.themes import THEMES
from types import SimpleNamespace
import os
import io
import base64
import json
from PIL import Image
import requests
import logging
from google import genai
from openai import OpenAI
from io import BytesIO
from core.postprocessing import PostProcessing
from google.genai import types
from typing import Dict, Any, Optional
from core.themes import THEMES_PRESETS_MIN, DEFAULTS
from core.utils import Utility

STRENGTH_INDEX = {"Light": 0, "Medium": 1, "Strong": 2}
NAPKIN_TEMPLATE = Utility.load_template()

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


class _SafeDict(dict):
    def __missing__(self, key):
        return ""


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
    def _safe_clean(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            if v is None:
                out[k] = ""
            elif isinstance(v, (list, tuple, set)):
                out[k] = ", ".join(map(str, v))
            else:
                out[k] = str(v)
        return out

    @staticmethod
    def _apply_design_overrides(
        base: Dict[str, Any], design: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        design may contain up to 5 user options:
        - color_palette: dict with 'base' and/or 'accent' (or a string)
        - pattern: string -> maps to background_library and (if empty) background_treatment
        - motif: string -> overrides motif
        - style: string -> overrides illustration_style
        - finish: string -> overrides finish_spec; if it mentions foil/metal, also metallic_finish
        """
        if not design:
            return base

        if "motif" in design and design["motif"]:
            base["motif"] = design["motif"]

        if "style" in design and design["style"]:
            base["illustration_style"] = design["style"]

        if "pattern" in design and design["pattern"]:
            base["background_library"] = design["pattern"]
            # fall back to same text as background_treatment if not explicitly set
            base.setdefault("background_treatment", design["pattern"])

        if "color_palette" in design and design["color_palette"]:
            cp = design["color_palette"]
            if isinstance(cp, dict):
                if cp.get("base"):
                    base["base_tones"] = cp["base"]
                if cp.get("accent"):
                    base["accent_colors"] = cp["accent"]
            else:
                # if string, put everything into base_tones
                base["base_tones"] = cp

        if "finish" in design and design["finish"]:
            fin = str(design["finish"])
            base["finish_spec"] = fin
            if any(word in fin.lower() for word in ("foil", "metal", "gold", "silver")):
                base["metallic_finish"] = fin

        return base

    @staticmethod
    def build_napkin_prompt(
        theme_key: str, extra: str = "", design: Optional[Dict[str, Any]] = None
    ) -> str:
        # start from global defaults, then theme preset
        if theme_key not in THEMES_PRESETS_MIN:
            raise KeyError(f"Unknown theme: {theme_key}")

        base = {**DEFAULTS, **THEMES_PRESETS_MIN[theme_key]}
        base["theme_label"] = theme_key
        base["extra"] = (extra or "").strip() or "—"

        base = Imagine._apply_design_overrides(base, design)

        # finalize
        text = NAPKIN_TEMPLATE.format_map(_SafeDict(Imagine._safe_clean(base)))
        # collapse whitespace to keep prompt tidy
        return " ".join(text.split())

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
    def image_to_base64(img_path):
        """Convert image to base64 for inline display."""
        with open(img_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")

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

    @staticmethod
    def save_image_with_metadata(
        img,
        save_path: str,
        combination: dict,
        json_path: str = "outputs/image_metadata.json",
    ):
        """
        Save image to disk and store its combination metadata in a JSON file.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)

        # Load existing metadata if present
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Save metadata using image path as key
        metadata[save_path] = combination

        # Write back to JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


class Generate:
    def __init__(
        self,
    ):
        pass

    def generate_with_openai(self, final_prompt: str, model_name: str = "dall-e-3"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {
                "status": False,
                "type": "Key Error",
                "msg": "GEMINI_API_KEY is not set.",
            }
        client = OpenAI(api_key=api_key) if api_key else None
        gen_kwargs = {
            "model": model_name,
            "prompt": final_prompt,
            "size": "1024x1024",
            "quality": "hd",
        }
        try:
            resp = client.images.generate(**gen_kwargs)
        except Exception as e:
            logger.error(f"Error Occurred while generating:{e}")
            return None, None
        if not resp or not getattr(resp, "data", None):
            return None, None
        else:
            for i, item in enumerate(resp.data, start=1):
                if getattr(item, "b64_json", None):
                    img = Imagine.b64_to_image(item.b64_json)
                elif getattr(item, "url", None):
                    try:
                        logger.info("Fetching image from URL")
                        resp = requests.get(item.url, timeout=10)
                        resp.raise_for_status()
                        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                    except Exception as e:
                        logger.warning(f"Could not fetch image from URL: {e}")
                        return None, None
                else:
                    logger.info(f"Result {i}: Unrecognized response format.")
                    return None, None
                if img is not None:
                    logger.info("Image generated successfully")
                    low, med, high = PostProcessing.apply_post_processing(img)
                    return img, {"low": low, "medium": med, "high": high}

    def generate_with_gemini(
        self,
        prompt: str,
    ):
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            return {
                "status": False,
                "type": "Key Error",
                "msg": "GEMINI_API_KEY is not set.",
            }
        client_gemini = genai.Client(api_key=GEMINI_API_KEY)
        """
        Calls the image model and returns a tuple of (original_image, enhanced_variants).
        enhanced_variants is a dict with keys low/medium/high or None on failure.
        """
        try:
            # --- Gemini image generation ---
            resp = client_gemini.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[prompt],
            )
            candidates = getattr(resp, "candidates", []) or []
            for candidate in candidates:
                parts = getattr(
                    candidate, "content", getattr(candidate, "contents", None)
                )
                parts = getattr(parts, "parts", []) if parts is not None else []
                for part in parts:
                    if part.inline_data is None:
                        continue
                    img = Image.open(BytesIO(part.inline_data.data)).convert("RGBA")
                    low, medium, high = PostProcessing.apply_post_processing(img)
                    return img, {"low": low, "medium": medium, "high": high}
            return None, None

        except Exception as e:
            logger.warning(f"Could not generate image: {e}")
            return None, None

    def generate_mock_image(
        self, index: int, folder: str = "outputs/now", count: int = 3
    ) -> tuple[str, Dict]:
        """
        Function to retun mock images to test the UI instead of generating images repeatedly
        """
        image_files = sorted(
            [
                f
                for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ],
            key=lambda x: os.path.getmtime(os.path.join(folder, x)),
            reverse=True,
        )[:count]
        index = index - 1
        img_path = os.path.join(folder, image_files[index])
        img = Image.open(img_path).convert("RGBA")
        if img is not None:
            low, medium, high = PostProcessing.apply_post_processing(img)
            return img, {"low": low, "medium": medium, "high": high}
        return None, None


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
