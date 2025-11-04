import os
import io
import logging
import re
import json
import time
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
from pathlib import Path
from openai import OpenAI
from typing import Dict, Any
from core.utils import Utility
from core.postprocessing import PostProcessing
from core.model import Imagine, Edit, Generate
from core.options import Options
from core.llm_combiner import LLMCombiner, GeminiClient
from google import genai
from datetime import datetime

# from termcolor import cprint

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
    page_title="Imagine - Premium Plate Generator",
    page_icon="assets/imagine_icon.svg",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 0% 0%, rgba(236, 243, 255, 0.95), rgba(255, 255, 255, 0.95));
    }
    .studio-card {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.83);
        border: 1px solid rgba(110, 125, 245, 0.08);
        # box-shadow: 0 18px 42px rgba(34, 53, 131, 0.08);
    }
    .studio-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        background: #ecf0ff;
        color: #4f5cd1;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
        
    }
    .prompt-box {
        background: rgba(241, 244, 255, 0.85);
        border: 1px solid rgba(79, 92, 209, 0.12);
        padding: 0.85rem 1rem;
        border-radius: 14px;
        margin-bottom:1rem;
        # box-shadow: 0 18px 42px rgba(34, 53, 131, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# st.title("🎨 Luma")
# st.header("AI-Powered Design for Paperware")
# st.subheader(
#     "Transform your paperware manufacturing with intelligent design automation"
# )
ICON = """
<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" viewBox="0 0 64 64" fill="none">
  <path d="M32 6 c-2.4 7.3-8.7 13.6-16 16 c7.3 2.4 13.6 8.7 16 16 c2.4-7.3 8.7-13.6 16-16 c-7.3-2.4-13.6-8.7-16-16z"
        fill="none" stroke="#7C3AED" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M50 10 v6 M47 13 h6" stroke="#7C3AED" stroke-width="6" stroke-linecap="round"/>
  <path d="M14 44 v6 M11 47 h6" stroke="#7C3AED" stroke-width="6" stroke-linecap="round"/>
</svg>
"""
st.markdown(
    f"""
    <h1 style='text-align: center;'>{ICON}Luma</h1>
    <h2 style='text-align: center;'>AI-Powered Design for Paperware</h2>
    <h4 style='text-align: center; color: gray; margin-bottom:3%;'>
        Transform your paperware manufacturing with intelligent design automation
    </h4>
    """,
    unsafe_allow_html=True,
)

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
def set_org(path: str):
    json_path = "outputs/image_metadata.json"
    with Image.open(path) as im:
        original = im.convert("RGBA").copy()

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    low, medium, high = PostProcessing.apply_post_processing(original)
    prompt = f"Generate a one-line rationale for the image named '{Path(path).name}'"
    data = metadata.get(path, {})
    combo = {
        "color_palette": data.get("color_palette"),
        "pattern": data.get("pattern"),
        "motif": data.get("motif"),
        "style": data.get("style"),
        "finish": data.get("finish"),
    }
    rationale = data.get("rationale")
    st.session_state.image_sets = [
        {
            "original": original,
            "enhanced": {"low": low, "medium": medium, "high": high},
            "edited": None,
            "combo": {**combo, "rationale": rationale},
            "prompt": prompt,
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


def remove_metadata_entry(image_path: str, json_path: str):
    """Remove a specific image entry from metadata.json."""
    if not os.path.exists(json_path):
        return

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if image_path in metadata:
        del metadata[image_path]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def delete_image(path: str, img_path: str):
    """Delete image file from disk (safety checks) and rerun."""
    try:
        json_path = "outputs/image_metadata.json"
        output_folder = "outputs/now"
        # safety: only delete inside outputs/now and image extensions
        allowed = (".png", ".jpg", ".jpeg", ".webp")
        if not path.startswith(os.path.abspath(output_folder)):
            st.warning("Refusing to delete outside the allowed folder.")
            return
        if not path.lower().endswith(allowed):
            st.warning("Refusing to delete non-image file.")
            return
        if os.path.exists(path):
            os.remove(path)
            remove_metadata_entry(img_path, json_path)
            st.toast("Image deleted.", icon="🗑️")
            logger.info(f"Image Deleted Successfully")
        else:
            st.info("File already removed.")
    except Exception as e:
        st.error(f"Could not delete image: {e}")
        logger.error(f"Could not delete image: {e}")


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
    # "🍔 Backyard BBQs / Cookouts",
    # "🏊‍♂️ Pool parties",
    # "🐣 Easter brunches",
    "🎃 Halloween parties",
    "🎉 New Year’s brunch",
    # "💼 Farewell or promotion parties at work",
]
options = Options()
# st.markdown(
#     """
#         <div class="studio-card">
#             <span class="studio-pill">Smart combinations</span>
#             <span class="studio-pill">Model-agnostic workflow</span>
#             <h3 style="margin-top: 0.4rem;">Design-ready prompts tailored to your generation process.</h3>
#             <p style="margin-bottom: 0.2rem;">
#                 Blend palettes, patterns, motifs, and finishes into polished briefs that align with your theme.
#                 Compare original and our enhanced version, download easily with one click.
#             </p>
#         </div>
#         """,
#     unsafe_allow_html=True,
# )
# hero_col, info_col = st.columns([3, 2], vertical_alignment="center")
# with hero_col:
#     st.markdown(
#         """
#         <div class="studio-card">
#             <span class="studio-pill">Smart combinations</span>
#             <span class="studio-pill">Model-agnostic workflow</span>
#             <h3 style="margin-top: 0.4rem;">Design-ready prompts tailored to your generation process.</h3>
#             <p style="margin-bottom: 0.2rem;">
#                 Blend palettes, patterns, motifs, and finishes into polished briefs that align with your theme.
#                 Compare original and our enhanced version, download easily with one click.
#             </p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

# with info_col:
#     """"""
# with st.container(border=True):
#     metric_cols = st.columns(5)
#     metric_cols[0].metric("Color Palettes", len(options.color_palettes))
#     metric_cols[1].metric("Pattern", len(options.patterns))
#     metric_cols[2].metric("Motif", len(options.motifs))
#     metric_cols[3].metric("Style", len(options.themes))
#     metric_cols[4].metric("Finishes", len(options.finishes))
#     st.markdown(
#         "<div class='prompt-box'>Need to seed ideas quickly? Mix & match options below and let the combiner propose elevated pairings.</div>",
#         unsafe_allow_html=True,
#     )

st.markdown(
    """
    <style>
    div[data-baseweb="select"] {
        width: fit-content !important;
        min-width: 240px;
    }
    div[data-baseweb="select"] > div {
        # background: radial-gradient(circle at 0% 0%, rgba(220, 235, 255, 0.95), rgba(240, 225, 255, 0.95)) !important;
        background : white;
        border-radius: 8px;
        border: 1px solid #c3c8d4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        min-width:50%;
    }
    .stTextArea {
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
left, right = st.columns([6, 2], vertical_alignment="top")
with st.form("gen"):
    st.markdown("### Design Theme")
    theme_key = st.selectbox("Theme", themes, index=0, width="stretch")

    # st.markdown("### Render Settings")
    # option = st.selectbox(
    #     "Choose Enhancement Level:", ["🌙 Low", "🔆 Medium", "🌟 High"]
    # )
    option = "high"
    opt = Options()
    st.subheader("Design Options")
    st.badge(
        "Default will choose three best combination from the drop down",
        icon="🚨",
        color="blue",
    )
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        sel_palette = select_with_default(
            "Color Palette", list(opt.color_palettes), "sel_palette"
        )
    with r2:
        sel_pattern = select_with_default("Pattern", list(opt.patterns), "sel_pattern")
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
    st.markdown("### Extra Detail (Optional)")
    extra = st.text_area(
        "Add any small tweaks (e.g., butterflies, warmer gold, softer stripes)",
        height=80,
    )

    st.write("")
    submitted = st.form_submit_button(
        "Generate",
        width="content",
        icon="⚙️",
        help="click to generate Image",
    )
gallery = st.empty()

# with right:
#     output_folder = "outputs/now"
#     thumb_w = 115
#     if os.path.exists(output_folder):
#         # Get all image files (sorted by latest)
#         image_files = sorted(
#             [
#                 f
#                 for f in os.listdir(output_folder)
#                 if f.lower().endswith((".png", ".jpg", ".jpeg"))
#             ],
#             key=lambda x: os.path.getmtime(os.path.join(output_folder, x)),
#             reverse=True,
#         )[
#             :5
#         ]  # latest 5
#         with st.container(border=True):
#             st.subheader("🖼️ Previous Images")
#             for i, fname in enumerate(image_files):
#                 img_path = os.path.join(output_folder, fname)
#                 col_img, col_btns = st.columns(
#                     [1.5, 0.5]
#                 )  # left: image, right: buttons

#                 with col_img:
#                     try:
#                         st.image(Image.open(img_path), width=thumb_w)
#                     except Exception as e:
#                         st.warning(f"Could not open {fname}: {e}")

#                 with col_btns:
#                     st.button(
#                         "Use",
#                         key=f"use_{i}",
#                         on_click=set_org,
#                         args=(img_path,),
#                         width="stretch",
#                         help="Use",
#                     )
#                     st.button(
#                         "🗑️",
#                         key=f"del_{i}",
#                         on_click=delete_image,
#                         args=(os.path.abspath(img_path), img_path),
#                         width="stretch",
#                         help="Delete",
#                     )
#     else:
#         st.info("No previous images found yet.")


# =========================================================
# Submit handler
# =========================================================

if "combiner" not in st.session_state:
    st.session_state.combiner = LLMCombiner(llm_fn=gemini_client.gemini_call)


def _slug(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")


if submitted:
    gen = Generate()
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
            rationale = gemini_client.ask_gemini(user_combo)
            time.sleep(2)
            st.session_state["last_llm_combinations"] = [
                {**user_combo, "rationale": rationale}
            ]
            user_combo["rationale"] = rationale if rationale else ""
            designs_to_run = [user_combo]

        logger.info(f"Generating {len(designs_to_run)} prompt(s) & image(s).")
        os.makedirs("outputs/now", exist_ok=True)
        theme_slug = _slug(theme_key)

        with st.spinner("Generating Image"):
            # Optional: show progress if multiple
            progress = st.empty()
            n_total = len(designs_to_run)

            for idx, combo in enumerate(designs_to_run, start=1):
                prompt_design = _strip_defaults(combo)
                # final_prompt = Imagine.build_napkin_prompt(
                #     theme_key=theme_key, design=prompt_design, extra=extra
                # )
                final_prompt = Imagine.build_plate_prompt(combo)
                # cprint(final_prompt, "yellow")
                img, variants = gen.generate_with_gemini(final_prompt)
                # img, variants = gen.generate_with_openai(final_prompt)
                # img, variants = gen.generate_mock_image(index=idx)
                if img is None or variants is None:
                    st.info(f"No image returned for combo {idx}.")
                    continue

                # Save to disk (include combo index + short spec in name)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                spec_parts = [prompt_design.get(key, "default") for key in ATTR_KEYS]
                short_spec = _slug("-".join(spec_parts))[:60]
                filename = f"napkin_{theme_slug}_c{idx}_{stamp}_{short_spec}.png"
                save_path = os.path.join("outputs/now", filename)
                comb_dict = combo if combos else user_combo
                Imagine.save_image_with_metadata(
                    img=img, save_path=save_path, combination=comb_dict
                )

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

left, right = st.columns([6, 2], vertical_alignment="top")
with right:
    image_sets = st.session_state.get("image_sets", [])
    if image_sets:
        with st.container(border=True):
            st.subheader("Design")
            combo_labels = []
            for idx, combo_set in enumerate(image_sets):
                combo = combo_set.get("combo", {}) or {}
                label_bits = [
                    str(combo.get(key, "")).title()
                    for key in ("motif", "pattern")
                    if combo.get(key)
                ]
                hint = ", ".join(label_bits[:2])
                st.badge(f"Combination {idx+1 if len(image_sets) > 1 else ''}")
                st.markdown(
                    """
                    <style>
                    .kv-pair {
                        font-size: 14px;
                        margin-bottom: 4px;
                    }
                    .kv-label {
                        font-weight: 600;
                        color: #4B5563; /* cool gray */
                        text-transform: capitalize;
                    }
                    .kv-value {
                        color: #111827;
                        margin-left: 4px;
                    }
                    .divider {
                        height: 1px;
                        background-color: #E5E7EB;
                        margin: 8px 0 10px 0;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                for k, v in combo.items():
                    if k == "rationale":
                        continue
                    label = k.replace("_", " ")
                    val = v.title()
                    st.markdown(
                        f"<div class='kv-pair'><span class='kv-label'>{label}:</span> "
                        f"<span class='kv-value'>{val}</span></div>"
                        f"<div class='divider'></div>",
                        unsafe_allow_html=True,
                    )
                combo_labels.append(f"Combination {idx + 1}")

            default_idx = min(
                st.session_state.get("selected_combo_index", 0),
                len(image_sets) - 1,
            )
            # combo_index = st.selectbox(
            #     "Select the combination",
            #     options=list(range(len(image_sets))),
            #     index=default_idx,
            #     format_func=lambda i: combo_labels[i],
            #     key="combo_selector",
            # )
            # st.session_state.selected_combo_index = combo_index
            # selected_combo = image_sets[combo_index].get("combo", {}) or {}
            # rationale = selected_combo.get("rationale")
            # if rationale:
            #     st.caption(f"Why it works: {rationale}")

            # enhancement_options = ["Low", "Medium", "High"]
            # combo_levels = st.session_state.get("combo_enhancement_levels", {})
            # default_level = combo_levels.get(
            #     combo_index, st.session_state.get("enhancement_level", "Low")
            # )
            # if default_level not in enhancement_options:
            #     default_level = "Low"
            # select_key = f"enhancement_level_combo_{combo_index}"
            # selected_level = st.selectbox(
            #     "Select the enhancement for preview",
            #     enhancement_options,
            #     key=select_key,
            # )
            # combo_levels[combo_index] = selected_level
            # st.session_state.enhancement_level = selected_level

            # source_map = {
            #     "Original": ("original", None),
            #     "Low Enhanced": ("enhanced", "low"),
            #     "Medium Enhanced": ("enhanced", "medium"),
            #     "High Enhanced": ("enhanced", "high"),
            # }
            # st.markdown("#### Prompt")
            # mask_file = st.selectbox(
            #     "Select the source image to edit",
            #     list(source_map.keys()),
            #     key="edit_base_choice",
            # )
            # st.caption(f"Selected Combination for edit is : {combo_index+1}")
            # base_kind, variant_key = source_map[mask_file]
            # if base_kind == "original":
            #     base_img = image_sets[combo_index]["original"]
            # else:
            #     base_img = image_sets[combo_index]["enhanced"][variant_key]

            # # optional: show a tiny thumbnail of the chosen base
            # # st.image(base_img, width=50)
            # edit_prompt = st.text_area(
            #     "Describe how to edit the image (e.g., 'make the background white, add subtle gold shimmer')",
            #     height=80,
            #     key="edit_prompt",
            #     placeholder="Type your edit prompt here...",
            # )
            # apply = st.button("Apply Edit", width="stretch")
            # if apply:
            #     status = "editing"
            #     if not edit_prompt.strip():
            #         st.warning("Please write an edit prompt.")
            #     else:
            #         try:
            #             edit = Edit()
            #             image_to_edit = base_img
            #             response = edit.edit_with_gemini(
            #                 base_img=image_to_edit, prompt=edit_prompt
            #             )
            #             with st.spinner("editing.."):
            #                 if response.get("status"):
            #                     eimg = response.get("images")["org"]
            #                     st.session_state.image_sets[combo_index][
            #                         "edited"
            #                     ] = eimg
            #                     st.success("Image Edited Successfully")
            #                 else:
            #                     st.error(f"Unable to edit, got {response.get('type')}")
            #         except Exception as e:
            #             logger.error(f"Error Occurred while editing: {e}")

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
                chosen_level = "Medium"
            chosen_variant_key = key_map[chosen_level]

            combo = image_set.get("combo", {}) or {}
            header = (
                f"Combination {idx + 1}"
                if len(image_sets) > 1
                else "Selected Combination"
            )
            st.markdown(f"**{header}**")

            details = [
                f"{key.replace('_', ' ').title()}: {combo.get(key)}"
                for key in ATTR_KEYS
                if combo.get(key) and combo.get(key) != "Default"
            ]
            rationale = combo.get("rationale")
            # if details:
            #     st.caption("; ".join(details))
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
                    icon="⬇️",
                )

            with cols[1]:
                enhanced_img = image_set["enhanced"].get(
                    chosen_variant_key, image_set["enhanced"]["low"]
                )
                st.image(
                    enhanced_img,
                    caption=f"Enhanced",
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
                    icon="⬇️",
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
                        icon="⬇️",
                    )

            if idx < len(image_sets) - 1:
                st.divider()
    else:
        st.info("Generate an image to preview and download.")
