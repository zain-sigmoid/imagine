import os
import io
import requests
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
from openai import OpenAI
from typing import Dict, Any
from core.utils import Utility
from core.postprocessing import PostProcessing
from core.model import Imagine
from core.prompt_store import init_db, save_prompt, get_recent_prompts
from core.themes import THEMES_PRESETS_MIN

# =========================================================
# Setup
# =========================================================
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
        st.info("OPENAI_API_KEY is not set. Please add it and try again.")
    else:
        final_prompt = build_napkin_prompt(theme_key, extra)

        # Save & refresh recents
        save_prompt(final_prompt)
        st.session_state.recent_prompts = get_recent_prompts(limit=5)

        with st.spinner("Generating..."):
            model_name = "dall-e-3"
            gen_kwargs = {"model": model_name, "prompt": final_prompt, "size": size}
            if quality != "standard":
                gen_kwargs["quality"] = quality
            try:
                resp = client.images.generate(**gen_kwargs)
                # resp = Imagine.mock_response_from_file(path="napkin_enhanced.png")
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
                            print("Fetching image from URL:")
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
            # st.subheader("Edit your Image")
            level = st.selectbox(
                "Change Enhancement Level",
                ["Low", "Medium", "High"],
                key="enhancement_level",
            )
    else:
        """"""
with left:
    if has_imgs:
        c1, c2 = st.columns(2)

        with c1:
            org = st.session_state.images["org"]
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

        with c2:
            key_map = {"Low": "low", "Medium": "medium", "High": "high"}
            chosen_key = st.session_state.enhancement_level
            img = st.session_state.images[key_map[chosen_key]]

            st.image(
                img,
                caption=f"Enhanced ({st.session_state.enhancement_level})",
                width="stretch",
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button(
                "Download Enhanced",
                buf.getvalue(),
                file_name=f"napkin_{st.session_state.enhancement_level.lower()}_enhanced.png",
                mime="image/png",
                width="stretch",
            )
    else:
        st.info("Generate an image to preview and download.")
