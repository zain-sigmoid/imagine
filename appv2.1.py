import os
import io
import requests
import logging
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
from pathlib import Path
from openai import OpenAI
from typing import Dict, Any
from termcolor import cprint
from core.utils import Utility
from core.postprocessing import PostProcessing
from core.model import Imagine
from core.prompt_store import init_db, save_prompt, get_recent_prompts
from core.themes import THEMES_PRESETS_MIN
from google import genai
from google.genai import types
from io import BytesIO

# =========================================================
# Setup
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY) if API_KEY else None

st.set_page_config(
    page_title="Imagine - Premium Napkin Generator", page_icon="🎨", layout="wide"
)
st.title("🎨 Premium Paper Napkin — Theme-led Generator")
st.caption(
    "Pick a theme, choose render settings, (optionally) add extra art direction."
)
init_db()

if not API_KEY:
    st.warning("Set OPENAI_API_KEY in your environment or .env file.", icon="⚠️")

if "recent_prompts" not in st.session_state:
    st.session_state.recent_prompts = []
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

if "images" not in st.session_state:
    st.session_state.images = {}

if "enhancement_level" not in st.session_state:
    st.session_state.enhancement_level = "Low"
# st.session_state.recent_prompts = get_recent_prompts(limit=5)


# Load the string template with placeholders like {motif}, {background_treatment}, {extra}, etc.
NAPKIN_TEMPLATE = Utility.load_template()


# =========================================================
# Prompt builder helpers
# =========================================================
class _SafeDict(dict):
    def __missing__(self, key):
        return ""


def _safe_clean(d: Dict[str, Any]) -> Dict[str, str]:
    return {k: (str(v).strip() if v is not None else "") for k, v in d.items()}


def build_napkin_prompt(theme_key: str, extra: str) -> str:
    base = THEMES_PRESETS_MIN[theme_key].copy()
    base["theme_label"] = theme_key
    base["extra"] = extra.strip() if extra.strip() else "—"

    # Anti-crowding defaults
    base.setdefault("coverage_target", "30%")
    base.setdefault("min_margin_each_side", "25%")
    base.setdefault("max_floral_clusters", "3")  # total clusters cap
    base.setdefault("max_icon_count", "2")  # snowflakes/stars etc.
    final_prompt = " ".join(
        str(NAPKIN_TEMPLATE).format_map(_SafeDict(_safe_clean(base))).split()
    )
    return final_prompt


# =========================================================
# Layout: left (form/results) | right (recents)
# =========================================================
left, right = st.columns([6, 2], vertical_alignment="top")

with left:
    with st.form("gen"):
        st.markdown("### Design Theme")
        theme_key = st.selectbox("Theme", list(THEMES_PRESETS_MIN.keys()), index=0)

        st.markdown("### Render Settings")
        # Defaults come from theme preset; user can override here
        default_size = THEMES_PRESETS_MIN[theme_key]["api_size"]
        default_quality = THEMES_PRESETS_MIN[theme_key]["quality_hint"]

        r1, r2, r3 = st.columns(3)
        with r1:
            size = st.selectbox(
                "Size",
                ["1024x1024", "1024x1792", "1792x1024"],
                index=(
                    ["1024x1024", "1024x1792", "1792x1024"].index(default_size)
                    if default_size in ["1024x1024", "1024x1792", "1792x1024"]
                    else 0
                ),
            )
        with r2:
            quality = st.selectbox(
                "Quality",
                ["standard", "hd"],
                index=(0 if default_quality == "standard" else 1),
            )
        with r3:
            option = st.selectbox(
                "Choose Enhancement Level:", ["Low", "Medium", "High"]
            )

        st.markdown("### Extra Detail (optional)")
        extra = st.text_area(
            "Add any small tweaks (e.g., butterflies, warmer gold, softer stripes)",
            height=80,
        )

        submitted = st.form_submit_button("Generate", width="stretch")

    # Placeholder to render/clear results
    gallery = st.empty()

# Right: recent prompts (compact)
with right:
    """"""
    # st.subheader("Recent prompts")
    # try:
    #     scroll = st.container(height=320, border=False)
    # except TypeError:
    #     scroll = None

    # def render_recent_list(container):
    #     recents = st.session_state.recent_prompts
    #     if recents:
    #         for rp in recents:
    #             with container.container(border=True):
    #                 st.code(rp, language=None, wrap_lines=True, height=75)
    #     else:
    #         container.caption("No prompts yet.")

    # if scroll:
    #     render_recent_list(scroll)
    # else:
    #     st.caption("No prompts yet.")

# =========================================================
# Submit handler
# =========================================================
if submitted:
    if not client:
        logger.error("OPEN AI KEY Missing")
        st.info("OPENAI_API_KEY is not set. Please add it and try again.")
    else:
        logger.info("Building Prompt")
        final_prompt = build_napkin_prompt(theme_key, extra)

        # Save & refresh recents
        save_prompt(final_prompt)
        st.session_state.recent_prompts = get_recent_prompts(limit=5)

        logger.info("Generating Image")
        with st.spinner("Generating..."):
            model_name = "dall-e-3"
            gen_kwargs = {"model": model_name, "prompt": final_prompt, "size": size}
            if quality != "standard":
                gen_kwargs["quality"] = quality
            try:
                # resp = client.images.generate(**gen_kwargs)
                resp = Imagine.mock_response_from_file(path="demo/napkin_original.png")
            except Exception as e:
                st.info(f"Could not generate the image: {e}")
                resp = None

        if not resp or not getattr(resp, "data", None):
            st.info("No image returned. Try a different theme or simplify extras.")
        else:
            with gallery.container():
                for i, item in enumerate(resp.data, start=1):
                    if getattr(item, "b64_json", None):
                        img = Imagine.b64_to_image(item.b64_json)
                    elif getattr(item, "url", None):
                        try:
                            cprint("Fetching image from URL:", "yellow")
                            resp = requests.get(item.url, timeout=10)
                            resp.raise_for_status()
                            img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                        except Exception as e:
                            st.warning(f"Could not fetch image from URL: {e}")
                    else:
                        st.info(f"Result {i}: Unrecognized response format.")
                        img = None
                    if img is not None:
                        st.session_state.images["org"] = img
                        low, med, high = PostProcessing.apply_post_processing(img)
                        st.session_state.images["low"] = low
                        st.session_state.images["medium"] = med
                        st.session_state.images["high"] = high

left, right = st.columns([6, 2], vertical_alignment="top")
with right:
    has_imgs = all(
        k in st.session_state.images for k in ["org", "low", "medium", "high"]
    )
    if has_imgs:
        with st.container(border=True):
            st.subheader("Edit your Image")
            level = st.selectbox(
                "Select The Image",
                ["Low", "Medium", "High"],
                key="enhancement_level",
            )
            source_map = {
                "Original": "org",
                "Low Enhanced": "low",
                "Medium Enhanced": "medium",
                "High Enhanced": "high",
            }
            st.markdown("#### Prompt")
            mask_file = st.selectbox(
                "Select the source image to edit",
                list(source_map.keys()),
                key="edit_base_choice",
            )
            base_key = source_map[mask_file]
            base_img: Image.Image = st.session_state.images[base_key]

            # optional: show a tiny thumbnail of the chosen base
            # st.image(base_img, width=50)
            edit_prompt = st.text_area(
                "Describe how to edit the image (e.g., 'make the background white, add subtle gold shimmer')",
                height=80,
                key="edit_prompt",
                placeholder="Type your edit prompt here...",
            )
            apply = st.button("Apply Edit", width="stretch")
            if apply:
                if not edit_prompt.strip():
                    st.warning("Please write an edit prompt.")
                else:
                    # try:
                    #     api_key = os.environ.get("GEMINI_API_KEY")
                    #     if not api_key:
                    #         st.error("GEMINI_API_KEY is not set.")
                    #     else:
                    #         client = genai.Client(api_key=api_key)

                    #         # Build request: text + image
                    #         img_bytes = Imagine.pil_to_png_bytes(base_img)
                    #         # parts = [
                    #         #     types.Part.from_bytes(
                    #         #         data=img_bytes,
                    #         #         mime_type="image/png",
                    #         #     ),
                    #         # ]

                    #         # Call Gemini 2.5 Flash Image (preview name may still be required in some environments)
                    #         model_name = "gemini-2.5-flash-image-preview"  # or "gemini-2.5-flash-image-preview"
                    #         resp = client.models.generate_content(
                    #             model=model_name,
                    #             contents=[
                    #                 edit_prompt,
                    #                 types.Part.from_bytes(
                    #                     data=img_bytes, mime_type="image/png"
                    #                 ),  # or use this helper
                    #             ],
                    #         )

                    #         # Parse response: find first image part
                    #         edited_bytes = None
                    #         if resp and resp.candidates:
                    #             for part in resp.candidates[0].content.parts:
                    #                 b = Imagine.read_gemini_image_part(part)
                    #                 if b:
                    #                     edited_bytes = b
                    #                     break

                    #         if not edited_bytes:
                    #             st.error("Edit response missing image data.")
                    #         else:
                    #             edited_img = Image.open(BytesIO(edited_bytes)).convert(
                    #                 "RGBA"
                    #             )

                    #             # Update originals + rebuild your enhanced variants
                    #             st.session_state.images["org"] = edited_img
                    #             low, med, high = PostProcessing.apply_post_processing(
                    #                 edited_img
                    #             )
                    #             st.session_state.images["low"] = low
                    #             st.session_state.images["medium"] = med
                    #             st.session_state.images["high"] = high

                    #             st.success("Edit applied! Preview updated on the left.")
                    # try:
                    #     api_key = os.environ.get("OPENAI_API_KEY")
                    #     if not api_key:
                    #         st.error("OPENAI_API_KEY is not set.")
                    #     else:
                    #         size = Imagine.pick_openai_size_from_image(base_img)
                    #         files = {
                    #             "image": (
                    #                 "image.png",
                    #                 Imagine.pil_to_png_bytes(base_img),
                    #                 "image/png",
                    #             ),
                    #         }
                    #         data = {
                    #             "model": "dall-e-2",  # supports edits/inpainting
                    #             "prompt": edit_prompt,
                    #             "size": size,
                    #         }
                    #         cprint(f"Editing with prompt: {edit_prompt}", "green")
                    #         cprint("Sending edit request to OpenAI...", "green")
                    #         resp = requests.post(
                    #             "https://api.openai.com/v1/images/edits",
                    #             headers={"Authorization": f"Bearer {api_key}"},
                    #             data=data,
                    #             files=files,
                    #             timeout=90,
                    #         )
                    #         resp.raise_for_status()
                    #         cprint("Edit response received.", "cyan")

                    #         payload = resp.json()
                    #         if not payload.get("data"):
                    #             st.error("No image returned from edit.")
                    #         else:
                    #             out = payload.get("data", [{}])[0]
                    #             eimg = None
                    #             if out.get("b64_json"):
                    #                 # if you forced response_format="b64_json"
                    #                 eimg = Imagine.b64_to_image(out["b64_json"])
                    #             elif out.get("url"):
                    #                 # fallback: download from signed URL
                    #                 print("Fetching edited image from URL...")
                    #                 r = requests.get(out["url"], timeout=30)
                    #                 r.raise_for_status()
                    #                 eimg = Image.open(io.BytesIO(r.content)).convert(
                    #                     "RGBA"
                    #                 )
                    #                 # st.image(eimg, caption="Edited Image", width=200)
                    #             if eimg is None:
                    #                 st.error("Edit failed: no image returned.")
                    #             else:
                    #                 # store + rebuild enhanced variants
                    #                 # st.session_state.images["org"] = img
                    #                 st.session_state.images["edited"] = eimg
                    #                 st.success(
                    #                     "Edit applied! Preview updated on the left."
                    #                 )

                    # except requests.HTTPError as e:
                    #     st.error(
                    #         f"Image edit failed: {e.response.text if e.response is not None else e}"
                    #     )
                    # except Exception as e:
                    #     st.error(f"Unexpected error during edit: {e}")
                    print("sunflower" in edit_prompt)
                    if "sunflower" in edit_prompt:
                        demo_image = "banana_e2.jpeg"
                    else:
                        demo_image = "nano_banana_edited.jpeg"

                    eimg = Image.open(f"demo/{demo_image}")
                    st.session_state.images["edited"] = eimg

    else:
        """"""
with left:
    if has_imgs:
        org = st.session_state.images["org"]
        key_map = {"Low": "low", "Medium": "medium", "High": "high"}
        chosen_key = st.session_state.get("enhancement_level", "Low")
        enhanced = st.session_state.images[key_map[chosen_key]]
        edited = st.session_state.images.get("edited", None)

        cols = st.columns(3 if edited is not None else 2)

        # Original
        with cols[0]:
            st.image(org, caption="Original", width="stretch")
            bufo = io.BytesIO()
            org.save(bufo, format="PNG")
            st.download_button(
                "Download Original",
                bufo.getvalue(),
                file_name="napkin_original.png",
                mime="image/png",
                width="stretch",
            )

        # Enhanced
        with cols[1]:
            st.image(enhanced, caption=f"Enhanced ({chosen_key})", width="stretch")
            buf = io.BytesIO()
            enhanced.save(buf, format="PNG")
            st.download_button(
                "Download Enhanced",
                buf.getvalue(),
                file_name=f"napkin_{chosen_key.lower()}_enhanced.png",
                mime="image/png",
                width="stretch",
            )

        # Edited (only if present)
        if edited is not None:
            with cols[2]:
                st.image(edited, caption="Edited", width="stretch")
                buf2 = io.BytesIO()
                edited.save(buf2, format="PNG")
                st.download_button(
                    "Download Edited",
                    buf2.getvalue(),
                    file_name="napkin_edited.png",
                    mime="image/png",
                    width="stretch",
                )
        # c1, c2 = st.columns(2)

        # with c1:
        #     org = st.session_state.images["org"]
        #     st.image(org, caption="Original", width="stretch")
        #     bufo = io.BytesIO()
        #     org.save(bufo, format="PNG")
        #     st.download_button(
        #         "Download Original",
        #         bufo.getvalue(),
        #         file_name="napkin_original.png",
        #         mime="image/png",
        #         width="stretch",
        #     )

        # with c2:
        #     key_map = {"Low": "low", "Medium": "medium", "High": "high"}
        #     chosen_key = st.session_state.enhancement_level
        #     img = st.session_state.images[key_map[chosen_key]]

        #     st.image(
        #         img,
        #         caption=f"Enhanced ({st.session_state.enhancement_level})",
        #         width="stretch",
        #     )
        #     buf = io.BytesIO()
        #     img.save(buf, format="PNG")
        #     st.download_button(
        #         "Download Enhanced",
        #         buf.getvalue(),
        #         file_name=f"napkin_{st.session_state.enhancement_level.lower()}_enhanced.png",
        #         mime="image/png",
        #         width="stretch",
        #     )
    else:
        st.info("Generate an image to preview and download.")
