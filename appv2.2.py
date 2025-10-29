import os
import io
import requests
import logging
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
from pathlib import Path
from openai import OpenAI
from typing import Dict, Any, Optional
from termcolor import cprint
from core.utils import Utility
from core.postprocessing import PostProcessing
from core.model import Imagine
from core.prompt_store import init_db, save_prompt, get_recent_prompts
from core.themes import THEMES_PRESETS_MIN, DEFAULTS
from core.options import Options
from core.llm_combiner import LLMCombiner, GeminiClient
from google import genai
from google.genai import types
from io import BytesIO
from datetime import datetime
import pandas as pd
from rich import print as rprint

# =========================================================
# Setup
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY) if API_KEY else None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client_gemini = genai.Client(api_key=GEMINI_API_KEY)
gemini_client = GeminiClient()

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

if "design" not in st.session_state:
    st.session_state.design = {}

if "selections" not in st.session_state:
    st.session_state.selections = {}

st.session_state.setdefault("generated_images", [])
st.session_state.setdefault("processed_images", [])
st.session_state["generated_images"].clear()
st.session_state["processed_images"].clear()
# st.session_state.recent_prompts = get_recent_prompts(limit=5)


# Load the string template with placeholders like {motif}, {background_treatment}, {extra}, etc.
NAPKIN_TEMPLATE = Utility.load_template()


# =========================================================
# Prompt builder helpers
# =========================================================
class _SafeDict(dict):
    def __missing__(self, key):
        return ""


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


def build_napkin_prompt(
    theme_key: str, extra: str = "", design: Optional[Dict[str, Any]] = None
) -> str:
    # start from global defaults, then theme preset
    if theme_key not in THEMES_PRESETS_MIN:
        raise KeyError(f"Unknown theme: {theme_key}")

    base = {**DEFAULTS, **THEMES_PRESETS_MIN[theme_key]}
    base["theme_label"] = theme_key
    base["extra"] = (extra or "").strip() or "—"

    base = _apply_design_overrides(base, design)

    # finalize
    text = NAPKIN_TEMPLATE.format_map(_SafeDict(_safe_clean(base)))
    # collapse whitespace to keep prompt tidy
    return " ".join(text.split())


def set_org(path: str):
    with Image.open(path) as im:
        im = im.convert("RGBA")
        st.session_state.images["org"] = im.copy()
        low, med, high = PostProcessing.apply_post_processing(im)
        st.session_state.images["low"] = low
        st.session_state.images["medium"] = med
        st.session_state.images["high"] = high
        if st.session_state.images.get("edited", None):
            st.session_state.images["edited"] = None


def select_with_default(label: str, values: list[str], key: str):
    """
    Show a selectbox with 'Default' as first option, values title-cased for display,
    but return the RAW value (not title-cased). If user picks Default → returns 'Default'.
    """
    display = ["Default"] + [v.title() for v in values]
    backmap = {"Default": "Default", **{v.title(): v for v in values}}
    chosen_disp = st.selectbox(label, display, index=0, key=key)
    return backmap[chosen_disp]


def any_default(d: dict) -> bool:
    return any(isinstance(v, str) and v.lower() == "default" for v in d.values())


def update_selection(name, key):
    st.session_state.selections[name] = st.session_state[key]


# =========================================================
# Layout: left (form/results) | right (recents)
# =========================================================
themes = [
    "Backyard BBQs / Cookouts",
    "Pool parties",
    "Easter brunches",
    "Halloween parties",
    "New Year’s brunch",
    "Farewell or promotion parties at work",
]
left, right = st.columns([6, 2], vertical_alignment="top")
with left:
    with st.form("gen"):
        st.markdown("### Design Theme")
        theme_key = st.selectbox("Theme", themes, index=0)

        st.markdown("### Render Settings")
        option = st.selectbox("Choose Enhancement Level:", ["Low", "Medium", "High"])
        opt = Options()
        st.subheader("Design Options")
        st.badge(
            "Default will choose three best combination from the drop down",
            icon="🚨",
            color="gray",
        )
        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            select_options = ["Default"] + [k.title() for k in opt.color_palettes]
            sel_palette = st.selectbox(
                "Color Palette", select_options, index=0, key="sel_palette"
            )
        with r2:
            select_options = ["Default"] + [k.title() for k in opt.patterns]
            sel_pattern = st.selectbox(
                "Pattern", select_options, index=0, key="sel_pattern"
            )
        with r3:
            select_options = ["Default"] + [k.title() for k in opt.motifs]
            sel_motif = st.selectbox("Motif", select_options, index=0, key="sel_motif")
        with r4:
            select_options = ["Default"] + [k.title() for k in opt.themes]
            sel_theme = st.selectbox("Style", select_options, index=0, key="sel_style")
        with r5:
            select_options = ["Default"] + [k.title() for k in opt.finishes]
            sel_finish = st.selectbox(
                "Finish", select_options, index=0, key="sel_finish"
            )

        selections = {
            "color_palette": sel_palette,
            "pattern": sel_pattern,
            "motif": sel_motif,
            "style": sel_theme,
            "finish": sel_finish,
        }
        st.session_state.selections = selections
        catalog = {
            "color_palette": list(opt.color_palettes),
            "pattern": list(opt.patterns),
            "motif": list(opt.motifs),
            "style": list(opt.themes),
            "finish": list(opt.finishes),
        }

        st.write("")  # small spacer
        disabled = any_default(st.session_state.selections)
        st.markdown("### Extra Detail (optional)")
        extra = st.text_area(
            "Add any small tweaks (e.g., butterflies, warmer gold, softer stripes)",
            height=80,
        )

        submitted = st.form_submit_button("Generate", width="stretch")
    gallery = st.empty()

with right:
    output_folder = "outputs/now"
    thumb_w = 120
    if os.path.exists(output_folder):
        # Get all image files (sorted by latest)
        image_files = sorted(
            [
                f
                for f in os.listdir(output_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ],
            key=lambda x: os.path.getmtime(os.path.join(output_folder, x)),
            reverse=True,
        )[
            :5
        ]  # latest 5
        with st.container():
            st.subheader("🖼️ Previous Images")
            # Render in rows, 2 columns per row
            for i in range(0, len(image_files), 2):
                cols = st.columns(2)
                # First image in the row
                img_path_1 = os.path.join(output_folder, image_files[i])
                with cols[0]:
                    st.image(Image.open(img_path_1), width=thumb_w)
                    st.button(
                        "Use", key=f"use_{i}", on_click=set_org, args=(img_path_1,)
                    )

                if i + 1 < len(image_files):
                    img_path_2 = os.path.join(output_folder, image_files[i + 1])
                    with cols[1]:
                        st.image(Image.open(img_path_2), width=thumb_w)
                        st.button(
                            "Use",
                            key=f"use_{i+1}",
                            on_click=set_org,
                            args=(img_path_2,),
                        )
    else:
        st.info("No previous images found yet.")


# =========================================================
# Submit handler
# =========================================================

if "combiner" not in st.session_state:
    st.session_state.combiner = LLMCombiner(llm_fn=gemini_client.gemini_call)

# If Top-3 button was pressed
# if "gen_top3" in locals() and gen_top3:
#     with st.spinner("Generating Combinations"):
#         combos = st.session_state.combiner.generate(selections, catalog)
#         st.session_state["last_llm_combinations"] = combos
#         st.rerun()


def _slug(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")


def _gen_one_image(prompt: str):
    """
    Calls your image model and returns a PIL.Image or None.
    Keeps your current dual-path parsing (b64/url) for compatibility.
    """
    try:
        # --- Gemini image generation ---
        resp = client_gemini.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
        )
        for part in resp.candidates[0].content.parts:
            if part.inline_data is not None:
                img = Image.open(BytesIO(part.inline_data.data))
                if img is not None:
                    st.session_state["generated_images"].append(img)
                    low, med, high = PostProcessing.apply_post_processing(img)
                    process = {"low": low, "med": med, "high": high}
                    st.session_state["processed_images"].append(process)
                    os.makedirs("outputs/now", exist_ok=True)
                    filename = f"napkin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    save_path = os.path.join("outputs/now", filename)

                    # save
                    img.save(save_path)
        return img

    except Exception as e:
        st.warning(f"Could not generate image: {e}")
        return None


def _gen_mock_image(index: int):
    """
    Function to retun mock images to test the UI instead of generating images repeatedly
    """
    folder = "outputs/now"
    image_files = sorted(
        [
            f
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ],
        key=lambda x: os.path.getmtime(os.path.join(folder, x)),
        reverse=True,
    )[:3]
    index = index - 1
    print(image_files, len(image_files), index)
    img_path = os.path.join(folder, image_files[index])
    img = Image.open(img_path)
    if img is not None:
        st.session_state["generated_images"].append(img)
        low, med, high = PostProcessing.apply_post_processing(img)
        process = {"low": low, "med": med, "high": high}
        st.session_state["processed_images"].append(process)
    return img


if submitted:
    if not client:
        logger.error("OPEN AI KEY Missing")
        st.info("OPENAI_API_KEY is not set. Please add it and try again.")
    else:
        logger.info("Building Prompt")
        design = {
            "color_palette": sel_palette,
            "pattern": sel_pattern,
            "motif": sel_motif,
            "style": sel_theme,
            "finish": sel_finish,
        }
        st.session_state.design = design
        combos = {}
        if disabled:
            with st.spinner("Generating Combinations for Default"):
                combos = st.session_state.combiner.generate(selections, catalog)
                st.session_state["last_llm_combinations"] = combos

        if combos and isinstance(combos, list) and len(combos) == 3:
            designs_to_run = [
                {
                    "color_palette": c["color_palette"],
                    "pattern": c["pattern"],
                    "motif": c["motif"],
                    "style": c["style"],
                    "finish": c["finish"],
                }
                for c in combos
            ]
        else:
            designs_to_run = [design]

        logger.info(f"Generating {len(designs_to_run)} prompt(s) & image(s).")
        os.makedirs("outputs/now", exist_ok=True)
        theme_slug = _slug(theme_key)
        # final_prompt = build_napkin_prompt(
        #     theme_key=theme_key, design=design, extra=extra
        # )
        with st.spinner("Generating Image"):
            # Optional: show progress if multiple
            progress = st.empty()
            n_total = len(designs_to_run)

            for idx, dsn in enumerate(designs_to_run, start=1):
                rprint(f"ruuning for design:{dsn}")
                final_prompt = build_napkin_prompt(
                    theme_key=theme_key, design=dsn, extra=extra
                )

                # img = _gen_one_image(final_prompt)
                img = _gen_mock_image(idx)
                if img is None:
                    st.info(f"No image returned for combo {idx}.")
                    continue

                # Save to disk (include combo index + short spec in name)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                short_spec = _slug(
                    f"{dsn['color_palette']}-{dsn['pattern']}-{dsn['motif']}-{dsn['style']}-{dsn['finish']}"
                )[:60]
                filename = f"napkin_{theme_slug}_c{idx}_{stamp}_{short_spec}.png"
                save_path = os.path.join("outputs/now", filename)
                # img.save(save_path)
                progress.markdown(f"**Done {idx}/{n_total}**")

        st.toast("Generation complete.", icon="✅")
        # with st.spinner("Generating..."):
        #     model_name = "dall-e-3"
        #     gen_kwargs = {
        #         "model": model_name,
        #         "prompt": final_prompt,
        #         "size": "1024x1024",
        #         "quality": "hd",
        #     }
        #     try:
        #         # resp = client.images.generate(**gen_kwargs)
        #         resp = client_gemini.models.generate_content(
        #             model="gemini-2.5-flash-image",
        #             contents=[final_prompt],
        #             config=types.GenerateContentConfig(
        #                 response_modalities=["IMAGE"],
        #             ),
        #         )
        #         cprint(resp, "yellow")
        #         for part in resp.candidates[0].content.parts:
        #             if part.text is not None:
        #                 print(part.text)
        #             elif part.inline_data is not None:
        #                 img = Image.open(BytesIO(part.inline_data.data))
        #                 if img is not None:
        #                     st.session_state.images["org"] = img
        #                     low, med, high = PostProcessing.apply_post_processing(img)
        #                     st.session_state.images["low"] = low
        #                     st.session_state.images["medium"] = med
        #                     st.session_state.images["high"] = high
        #                     os.makedirs("outputs/now", exist_ok=True)
        #                     filename = (
        #                         f"napkin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        #                     )
        #                     save_path = os.path.join("outputs/now", filename)

        #                     # save
        #                     img.save(save_path)
        #                     st.success("Image saved")
        #         # resp = Imagine.mock_response_from_file(path="demo/napkin_original.png")
        #     except Exception as e:
        #         st.info(f"Could not generate the image")
        #         logger.error(f"Error Occurred while generating:{e}")
        #         resp = None

        # if not resp or not getattr(resp, "data", None):
        #     st.info("No image returned.")
        # else:
        #     with gallery.container():
        #         for i, item in enumerate(resp.data, start=1):
        #             if getattr(item, "b64_json", None):
        #                 img = Imagine.b64_to_image(item.b64_json)
        #             elif getattr(item, "url", None):
        #                 try:
        #                     cprint("Fetching image from URL:", "yellow")
        #                     resp = requests.get(item.url, timeout=10)
        #                     resp.raise_for_status()
        #                     img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        #                 except Exception as e:
        #                     st.warning(f"Could not fetch image from URL: {e}")
        #             else:
        #                 st.info(f"Result {i}: Unrecognized response format.")
        #                 img = None
        #             if img is not None:
        #                 st.session_state.images["org"] = img
        #                 low, med, high = PostProcessing.apply_post_processing(img)
        #                 st.session_state.images["low"] = low
        #                 st.session_state.images["medium"] = med
        #                 st.session_state.images["high"] = high
        #                 os.makedirs("outputs/now", exist_ok=True)
        #                 filename = (
        #                     f"napkin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        #                 )
        #                 save_path = os.path.join("outputs/now", filename)

        #                 # save
        #                 img.save(save_path)
        #                 st.success("Image saved")

left, right = st.columns([6, 2], vertical_alignment="top")
with right:
    # has_imgs = all(
    #     k in st.session_state.images for k in ["org", "low", "medium", "high"]
    # )
    length = len(st.session_state["generated_images"])
    has_imgs = length > 0
    if has_imgs:
        with st.container(border=True):
            st.subheader("Edit your Image")
            if length > 1:
                combo = st.selectbox(
                    "Select The Combination",
                    ["First", "Second", "Third"],
                    key="combination",
                )
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
            base_img: Image.Image = st.session_state["generated_images"][0]

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
                    if "sunflower" in edit_prompt:
                        demo_image = "banana_e2.jpeg"
                    else:
                        demo_image = "nano_banana_edited.jpeg"

                    eimg = Image.open(f"demo/{demo_image}")
                    st.session_state.images["edited"] = eimg

    else:
        """"""
with left:
    length = len(st.session_state["generated_images"])
    has_imgs = length > 0
    if has_imgs:
        for i in range(length):
            org = st.session_state["generated_images"][i]
            key_map = {"Low": "low", "Medium": "medium", "High": "high"}
            chosen_key = st.session_state.get("enhancement_level", "Low")
            enhanced = st.session_state["processed_images"][i][key_map[chosen_key]]
            edited = st.session_state.images.get("edited", None)
            st.write(st.session_state["last_llm_combinations"][i])
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
                    key=f"{i}_org",
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
                    key=f"{i}_enh",
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
                        key=f"{i}_edt",
                    )
    else:
        st.info("Generate an image to preview and download.")
