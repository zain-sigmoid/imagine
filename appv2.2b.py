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
from core.model import Imagine, Edit
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

if "enhancement_level" not in st.session_state:
    st.session_state.enhancement_level = "Low"

if "design" not in st.session_state:
    st.session_state.design = {}

if "selections" not in st.session_state:
    st.session_state.selections = {}

st.session_state.setdefault("image_sets", [])
st.session_state.setdefault("last_llm_combinations", [])
st.session_state.setdefault("selected_combo_index", 0)
st.session_state.setdefault("combo_enhancement_levels", {})
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
        original = im.convert("RGBA").copy()

    low, medium, high = PostProcessing.apply_post_processing(original)
    st.session_state.image_sets = [
        {
            "original": original,
            "enhanced": {"low": low, "medium": medium, "high": high},
            "edited": None,
            "combo": {"rationale": f"Loaded from {Path(path).name}"},
            "prompt": "",
        }
    ]
    st.session_state.last_llm_combinations = []
    st.session_state.selected_combo_index = 0
    st.session_state.combo_enhancement_levels = {
        0: st.session_state.get("enhancement_level", "Low")
    }
    st.session_state["enhancement_level_combo_0"] = st.session_state.get(
        "enhancement_level", "Low"
    )


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


ATTR_KEYS = ("color_palette", "pattern", "motif", "style", "finish")


def _strip_defaults(values: Dict[str, Any]) -> Dict[str, str]:
    cleaned: Dict[str, str] = {}
    for key in ATTR_KEYS:
        val = values.get(key)
        if isinstance(val, str) and val.strip() and val.strip().lower() != "default":
            cleaned[key] = val.strip()
    return cleaned


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
            sel_palette = select_with_default(
                "Color Palette", list(opt.color_palettes), "sel_palette"
            )
        with r2:
            sel_pattern = select_with_default(
                "Pattern", list(opt.patterns), "sel_pattern"
            )
        with r3:
            sel_motif = select_with_default("Motif", list(opt.motifs), "sel_motif")
        with r4:
            sel_theme = select_with_default("Style", list(opt.themes), "sel_style")
        with r5:
            sel_finish = select_with_default("Finish", list(opt.finishes), "sel_finish")

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
        has_default = any_default(st.session_state.selections)
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
            parts = getattr(candidate, "content", getattr(candidate, "contents", None))
            parts = getattr(parts, "parts", []) if parts is not None else []
            for part in parts:
                if part.inline_data is None:
                    continue
                img = Image.open(BytesIO(part.inline_data.data)).convert("RGBA")
                low, medium, high = PostProcessing.apply_post_processing(img)
                return img, {"low": low, "medium": medium, "high": high}
        return None, None

    except Exception as e:
        st.warning(f"Could not generate image: {e}")
        return None, None


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
    img = Image.open(img_path).convert("RGBA")
    if img is not None:
        low, medium, high = PostProcessing.apply_post_processing(img)
        return img, {"low": low, "medium": medium, "high": high}
    return None, None


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
        st.session_state.enhancement_level = option
        st.session_state.image_sets = []
        st.session_state.selected_combo_index = 0
        st.session_state.combo_enhancement_levels = {}

        combos: list[Dict[str, Any]] = []
        if has_default:
            with st.spinner("Generating Combinations for Default"):
                combos = st.session_state.combiner.generate(selections, catalog)
        if combos:
            st.session_state["last_llm_combinations"] = combos
            designs_to_run = combos
        else:
            user_combo = {k: design[k] for k in ATTR_KEYS}
            st.session_state["last_llm_combinations"] = [
                {**user_combo, "rationale": "User-selected combination"}
            ]
            designs_to_run = [user_combo]

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

            for idx, combo in enumerate(designs_to_run, start=1):
                prompt_design = _strip_defaults(combo)
                final_prompt = build_napkin_prompt(
                    theme_key=theme_key, design=prompt_design, extra=extra
                )

                img, variants = _gen_one_image(final_prompt)
                # img, variants = _gen_mock_image(idx)
                if img is None or variants is None:
                    st.info(f"No image returned for combo {idx}.")
                    continue

                # Save to disk (include combo index + short spec in name)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                spec_parts = [prompt_design.get(key, "default") for key in ATTR_KEYS]
                short_spec = _slug("-".join(spec_parts))[:60]
                filename = f"napkin_{theme_slug}_c{idx}_{stamp}_{short_spec}.png"
                save_path = os.path.join("outputs/now", filename)
                img.save(save_path)

                combo_idx = len(st.session_state.image_sets)
                st.session_state.image_sets.append(
                    {
                        "original": img,
                        "enhanced": variants,
                        "edited": None,
                        "combo": combo,
                        "prompt": final_prompt,
                        "saved_path": save_path,
                    }
                )
                st.session_state.combo_enhancement_levels[combo_idx] = option
                st.session_state[f"enhancement_level_combo_{combo_idx}"] = option
                progress.markdown(f"**Done {idx}/{n_total}**")

        if st.session_state.image_sets:
            st.toast("Generation complete.", icon="✅")
        else:
            st.warning(
                "No images were generated. Please adjust your prompt and try again."
            )
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
    image_sets = st.session_state.get("image_sets", [])
    if image_sets:
        with st.container(border=True):
            st.subheader("Edit your Image")
            combo_labels = []
            for idx, combo_set in enumerate(image_sets):
                combo = combo_set.get("combo", {}) or {}
                label_bits = [
                    str(combo.get(key, "")).title()
                    for key in ("motif", "pattern")
                    if combo.get(key)
                ]
                hint = ", ".join(label_bits[:2])
                combo_labels.append(f"Combination {idx + 1}")

            default_idx = min(
                st.session_state.get("selected_combo_index", 0),
                len(image_sets) - 1,
            )
            combo_index = st.selectbox(
                "Select the combination",
                options=list(range(len(image_sets))),
                index=default_idx,
                format_func=lambda i: combo_labels[i],
                key="combo_selector",
            )
            st.session_state.selected_combo_index = combo_index
            selected_combo = image_sets[combo_index].get("combo", {}) or {}
            rationale = selected_combo.get("rationale")
            # if rationale:
            #     st.caption(f"Why it works: {rationale}")

            enhancement_options = ["Low", "Medium", "High"]
            combo_levels = st.session_state.get("combo_enhancement_levels", {})
            default_level = combo_levels.get(
                combo_index, st.session_state.get("enhancement_level", "Low")
            )
            if default_level not in enhancement_options:
                default_level = "Low"
            select_key = f"enhancement_level_combo_{combo_index}"
            selected_level = st.selectbox(
                "Select the enhancement for preview",
                enhancement_options,
                key=select_key,
            )
            combo_levels[combo_index] = selected_level
            st.session_state.enhancement_level = selected_level

            source_map = {
                "Original": ("original", None),
                "Low Enhanced": ("enhanced", "low"),
                "Medium Enhanced": ("enhanced", "medium"),
                "High Enhanced": ("enhanced", "high"),
            }
            st.markdown("#### Prompt")
            mask_file = st.selectbox(
                "Select the source image to edit",
                list(source_map.keys()),
                key="edit_base_choice",
            )
            st.caption(f"Selected Combination for edit is : {combo_index+1}")
            base_kind, variant_key = source_map[mask_file]
            if base_kind == "original":
                base_img = image_sets[combo_index]["original"]
            else:
                base_img = image_sets[combo_index]["enhanced"][variant_key]

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
                status = "editing"
                if not edit_prompt.strip():
                    st.warning("Please write an edit prompt.")
                else:
                    try:
                        edit = Edit()
                        image_to_edit = base_img
                        response = edit.edit_with_gemini(
                            base_img=image_to_edit, prompt=edit_prompt
                        )
                        with st.spinner("editing.."):
                            if response.get("status"):
                                eimg = response.get("images")["org"]
                                st.session_state.image_sets[combo_index][
                                    "edited"
                                ] = eimg
                                st.success("Image Edited Successfully")
                            else:
                                st.error(f"Unable to edit, got {response.get('type')}")
                    except Exception as e:
                        logger.error(f"Error Occurred while editing: {e}")

    else:
        """"""
with left:
    image_sets = st.session_state.get("image_sets", [])
    if image_sets:
        combo_levels = st.session_state.get("combo_enhancement_levels", {})
        key_map = {"Low": "low", "Medium": "medium", "High": "high"}

        for idx, image_set in enumerate(image_sets):
            chosen_level = combo_levels.get(
                idx, st.session_state.get("enhancement_level", "Low")
            )
            if chosen_level not in key_map:
                chosen_level = "Low"
            chosen_variant_key = key_map[chosen_level]

            combo = image_set.get("combo", {}) or {}
            header = f"Combination {idx + 1}"
            st.markdown(f"**{header}**")

            details = [
                f"{key.replace('_', ' ').title()}: {combo.get(key)}"
                for key in ATTR_KEYS
                if combo.get(key) and combo.get(key) != "Default"
            ]
            rationale = combo.get("rationale")
            if details:
                st.caption("; ".join(details))
            if rationale:
                st.caption(f"Rationale: {rationale}")

            edited_img = image_set.get("edited")
            cols = st.columns(3 if edited_img else 2)

            with cols[0]:
                st.image(
                    image_set["original"],
                    caption="Original",
                    width="content",
                )
                bufo = io.BytesIO()
                image_set["original"].save(bufo, format="PNG")
                st.download_button(
                    "Download Original",
                    bufo.getvalue(),
                    file_name=f"napkin_combo{idx + 1}_original.png",
                    mime="image/png",
                    key=f"{idx}_org",
                )

            with cols[1]:
                enhanced_img = image_set["enhanced"].get(
                    chosen_variant_key, image_set["enhanced"]["low"]
                )
                st.image(
                    enhanced_img,
                    caption=f"Enhanced ({chosen_level})",
                    width="content",
                )
                buf = io.BytesIO()
                enhanced_img.save(buf, format="PNG")
                st.download_button(
                    "Download Enhanced",
                    buf.getvalue(),
                    file_name=f"napkin_combo{idx + 1}_{chosen_level.lower()}_enhanced.png",
                    mime="image/png",
                    key=f"{idx}_enh",
                )

            if edited_img:
                with cols[2]:
                    st.image(edited_img, caption="Edited", width="stretch")
                    buf2 = io.BytesIO()
                    edited_img.save(buf2, format="PNG")
                    st.download_button(
                        "Download Edited",
                        buf2.getvalue(),
                        file_name=f"napkin_combo{idx + 1}_edited.png",
                        mime="image/png",
                        key=f"{idx}_edt",
                    )

            if idx < len(image_sets) - 1:
                st.divider()
    else:
        st.info("Generate an image to preview and download.")
