import os
import io
import base64
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
from openai import OpenAI
from types import SimpleNamespace
import time
from core.themes import THEMES
from core.model import Imagine
from core.prompt_store import init_db, save_prompt, get_recent_prompts


# ---------- Setup ----------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY) if API_KEY else None

st.set_page_config(
    page_title="Imagine - Generate Images", page_icon="🎨", layout="wide"
)

st.title("🎨 Generate Images from Prompt")
st.caption("Prompt-based image generation")
init_db()

if not API_KEY:
    st.warning("Set OPENAI_API_KEY in your environment or .env file.", icon="⚠️")


if "recent_prompts" not in st.session_state:
    st.session_state.recent_prompts = []
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

st.session_state.recent_prompts = get_recent_prompts(limit=5)


def preview_text(txt: str, max_chars: int = 60) -> str:
    txt = " ".join((txt or "").split())
    return txt if len(txt) <= max_chars else txt[:max_chars].rstrip() + "......"


STRENGTH_INDEX = {"Light": 0, "Medium": 1, "Strong": 2}


# ---------- Layout: left=form/results, right=recents (smaller + scrollable) ----------
left, right = st.columns([6, 2], vertical_alignment="top")

with left:
    with st.form("gen"):
        prompt = st.text_area(
            "Describe your image",
            key="prompt",
            height=140,
            placeholder="e.g., an ultrarealistic photo of a cozy reading nook at sunrise, soft bokeh, 35mm film look",
        )
        prompt.join(st.session_state.prompt)
        col1, col2 = st.columns(2)
        with col1:
            size = st.selectbox(
                "Size", ["1024x1024", "1024x1792", "1792x1024"], index=0
            )
        with col2:
            quality = st.selectbox(
                "Quality",
                ["standard", "hd"],
                index=0,
                help="Use 'hd' if your plan/model supports it (may cost more).",
            )
        t1, t2 = st.columns([2, 2])
        with t1:
            theme_name = st.selectbox("Theme", list(THEMES.keys()), index=0)
        with t2:
            strength_label = st.radio(
                "Theme strength",
                ["Light", "Medium", "Strong"],
                index=1,
                horizontal=True,
            )

        submitted = st.form_submit_button("Generate", use_container_width=True)

    # Placeholder to render/clear results
    gallery = st.empty()

# --- Add to recents immediately and clear previous images ---
if submitted and prompt.strip():
    recents = st.session_state.recent_prompts
    if prompt in recents:
        recents.remove(prompt)
    recents.insert(0, prompt)
    st.session_state.recent_prompts = recents[:5]
    gallery.empty()


def mock_response_from_file(path: str):
    # Read and encode the image
    with open(path, "rb") as f:
        img_bytes = f.read()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")

    # Match OpenAI's response format: resp.data[0].b64_json
    fake_item = SimpleNamespace(b64_json=b64_str, url=None)
    fake_resp = SimpleNamespace(data=[fake_item])
    return fake_resp


# ---------- Right column: Recent prompts (compact, scrollable, no buttons) ----------
with right:
    st.subheader("Recent prompts")

    # Try Streamlit's scrollable container; fallback to CSS on older versions
    scroll = None
    try:
        scroll = st.container(height=320, border=False)  # smaller fixed height
    except TypeError:
        st.markdown(
            """
            <style>
              .recent-scroll { max-height: 320px; overflow-y: auto; padding-right: 8px; }
              .recent-card { border: 1px solid rgba(100,100,100,0.25); border-radius: 10px;
                             padding: 10px; margin-bottom: 8px; background: rgba(200,200,200,0.05); }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="recent-scroll">', unsafe_allow_html=True)

    def render_recent_list(container):
        if st.session_state.recent_prompts:
            for idx, rp in enumerate(st.session_state.recent_prompts, start=1):
                with container.container(border=True):
                    # st.markdown(f"{preview_text(rp)}")
                    st.code(
                        rp,
                        language=None,
                        wrap_lines=True,
                        height=75,
                    )
                    # with st.expander("View full"):
                    #     # Users can copy/paste from here
                    #     st.code(rp, language=None)
        else:
            container.caption("No prompts yet.")

    if scroll:
        render_recent_list(scroll)
    else:
        # Fallback HTML box list + expanders
        if st.session_state.recent_prompts:
            for idx, rp in enumerate(st.session_state.recent_prompts, start=1):
                st.markdown(
                    f'<div class="recent-card"><strong>{idx}.</strong> {preview_text(rp)}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("View full"):
                    st.code(rp, language=None)
        else:
            st.caption("No prompts yet.")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------- Helper ----------
def b64_to_image(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGBA")


# ---------- Generate (after recents rendered) ----------
if submitted:
    if not client:
        st.info(
            "OPENAI_API_KEY is not set. Please add it in your environment or .env and try again."
        )
    elif not prompt.strip():
        st.info("Please enter a prompt.")
    else:
        final_prompt = Imagine.build_final_prompt(prompt, theme_name, strength_label)
        save_prompt(prompt)  # or save_prompt(prompt, db_path=DB_PATH)
        st.session_state.recent_prompts = get_recent_prompts(limit=5)
        # with st.expander("Final prompt used"):
        #     st.code(final_prompt, language=None, wrap_lines=True, height=100)
        with st.spinner("Generating..."):
            model_name = "dall-e-3"  # or "gpt-image-1" if enabled for your org
            gen_kwargs = {"model": model_name, "prompt": final_prompt, "size": size}
            if quality != "standard":
                gen_kwargs["quality"] = quality
            try:
                resp = client.images.generate(**gen_kwargs)
                # time.sleep(2)  # simulate network delay
                # resp = Imagine.mock_response_from_file("generated_standrd.png")
            except Exception as e:
                msg = str(e)
                if "must be verified" in msg.lower() and "gpt-image-1" in msg:
                    st.info(
                        "Your organization needs to be verified to use image generation. Please verify in the OpenAI dashboard and try again."
                    )
                else:
                    st.info(f"Could not generate the image: {msg}")
                resp = None

        # Render results into the cleared placeholder
        if not resp or not getattr(resp, "data", None):
            st.info("No image returned. Try a simpler prompt or 1024x1024.")
        else:
            with gallery.container():
                for i, item in enumerate(resp.data, start=1):
                    if getattr(item, "b64_json", None):
                        img = Imagine.b64_to_image(item.b64_json)
                        st.image(
                            img, caption=f"Output {i}", width="stretch"
                        )  # <-- no deprecated arg
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        st.download_button(
                            label=f"Download Image",
                            data=buf.getvalue(),
                            file_name=f"dalle_{i}.png",
                            mime="image/png",
                            width="stretch",
                        )
                    elif getattr(item, "url", None):
                        st.image(
                            item.url, caption=f"Output {i}", width="stretch"
                        )  # <-- no deprecated arg
                        if hasattr(st, "link_button"):
                            st.link_button(
                                "See in new tab ↗", item.url, width="stretch"
                            )
                        else:
                            st.markdown(
                                f'<a href="{item.url}" target="_blank" rel="noopener noreferrer">See in new tab ↗</a>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info(f"Result {i}: Unrecognized response format.")
