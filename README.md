# Imagine
Imagine is a Streamlit workspace for creating premium paper napkin artwork concepts.
It combines LLM powered creative direction with automated image generation and a
post-processing pipeline that produces ready-to-share previews at several enhancement
levels.
## Highlights
- Guided UI that walks through picking a theme, render strength, and optional design overrides.
- Gemini powered combiner proposes up to three cohesive option sets whenever the user keeps
  attributes on their default values.
- Image generation hooks for OpenAI (DALL-E) and Google Gemini, plus a mock mode that reuses
  local sample images for demos.
- Post-processing stack (OpenCV + Pillow) that outputs Low, Medium, and High enhancement variants.
- Downloadable assets saved to `outputs/now` so the latest renders are always at hand.
## Repository layout
```
.
|-- app.py               # Original prompt-driven UI using OpenAI only
|-- appv2.2b.py          # Latest "Premium Napkin" experience with themes and combos
|-- appv2.2.py           # Earlier iteration of the premium napkin flow
|-- core/
|   |-- llm_combiner.py  # Gemini prompt builder + JSON validator for combo suggestions
|   |-- model.py         # Shared helpers for image generation and editing APIs
|   |-- options.py       # Master list of palettes, motifs, finishes, etc.
|   |-- postprocessing.py# Enhancement pipeline applied to every render
|   |-- prompt_store.py  # SQLite-backed prompt history (auto-creates ~/.imagine_app)
|   |-- themes.py        # Theme presets referenced by the prompt template
|   |-- utils.py         # Template loading and miscellaneous helpers
|-- demo/                # Sample images used by the mock generator and UI demo buttons
|-- outputs/now/         # Latest generated assets (created on first run)
|-- requirements.txt
|-- runtime.txt          # Reference Python version (3.10)
```
## Prerequisites
- Python 3.10 (matching `runtime.txt`)
- pip (or another Python package manager)
- OpenAI API key (`OPENAI_API_KEY`)
- Google Gemini API key (`GEMINI_API_KEY`)
> Tip: `python-dotenv` is bundled, so you can keep the API keys in a local `.env`.
## Installation
```bash
git clone https://github.com/zain-sigmoid/imagine.git
cd imagine
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Or, if you prefer conda/mamba, create a Python 3.10 environment first and then install the
requirements file inside it.
## Environment variables
| Name             | Required | Purpose                                  |
| ---------------- | -------- | ---------------------------------------- |
| `OPENAI_API_KEY` | Yes      | Authenticates any OpenAI image/edit calls |
| `GEMINI_API_KEY` | Yes      | Enables Gemini combo suggestions and image generation |

- `GOOGLE_APPLICATION_CREDENTIALS` if you have a service-account based Gemini setup.
Create a file called `.env` in the project root to keep secrets out of source control:
```
OPENAI_API_KEY=sk-***
GEMINI_API_KEY=AIza***
```
## Running locally
All variants are Streamlit apps. Use whichever flows you need:
- Latest premium napkin experience (recommended):
  ```bash
  streamlit run appv2.2b.py
  ```
- Alternate v2.2 flow (similar UX without per-combo enhancement persistence):
  ```bash
  streamlit run appv2.2.py
  ```
- Original minimal prompt-to-image UI:
  ```bash
  streamlit run app.py
  ```
Set a custom port when required:
```bash
streamlit run appv2.2b.py --server.port 8080
```
## Using the premium napkin UI
1. Pick a theme and enhancement level on the form’s first page.
2. Override any of the palette, pattern, motif, style, or finish options, or leave them as
   `Default` to let Gemini supply three cohesive combinations.
3. Provide optional extra art direction text and click **Generate**.
4. Review each combination: swap between Original, Enhanced (Low/Medium/High), and any edited version.
5. Download the PNG captures or apply further edits via the right-side panel.
### Mock image mode
Without external API calls you can still demo the UI:
- Drop three sample images into `outputs/now`.
- Keep the default `_gen_mock_image` helper enabled (as shipped) to recycle those files per combo.
## Persistent data
- Generated images save under `outputs/now` with timestamps and a combination summary slug.
- Prompt history lives in `~/.imagine_app/prompts.db`. Delete that file if you need a fresh history.
## Customising options
Edit `core/options.py` to adjust the palette, pattern, motif, style, and finish catalogs that
populate the dropdowns. The Gemini combiner automatically respects new entries, as long as the
attribute names remain the same.
Theme prompt templates live in `core/themes.py`, and the main wording used for image prompts comes
from `core/templates.yml` (loaded via `core.utils.Utility.load_template()`).
## Troubleshooting
- **401/403 errors**: double-check that both API keys are active and exported to the environment
  Streamlit inherits.
- **No images appear**: the default code path currently uses `_gen_mock_image` for testing.
  Switch to `_gen_one_image` in `appv2.2b.py` to call the live Gemini image endpoint.
- **Missing OpenCV bindings**: ensure the install step used `opencv-python-headless` from
  `requirements.txt`. On Apple Silicon you may need to reinstall with `pip install --no-binary opencv-python-headless opencv-python-headless`.
- **Permission errors on prompt DB**: verify the process can write to `~/.imagine_app`.
## Contributing
1. Fork and clone the repository.
2. Create a feature branch.
3. Run `streamlit run appv2.2b.py` to validate UI changes.
4. Submit a pull request with screenshots and a short summary of the change.