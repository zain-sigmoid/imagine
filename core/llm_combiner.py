from __future__ import annotations
import json
import random
import os
from typing import Optional
from google import genai
from typing import Any, Callable, Dict, List, Optional

Json = Dict[str, Any]
LLMFn = Callable[[str], str]  # (prompt) -> raw_text_response

PROMPT_TEMPLATE = """\
You are a senior surface-pattern designer asked to propose the **top 3 print-ready combinations** for a premium napkin.

## Inputs
### User selections (may include "Default")
{user_selections_json}

### Available options catalog (each list contains the allowed values for that attribute)
{catalog_json}

## Decision Rules (very important)
1) Respect all **non-default** user selections exactly (do not change them).
2) For each attribute marked **"Default"**, you must **choose** values from the corresponding catalog.
3) If **all** attributes are "Default": return your **3 best diverse** full combinations across all attributes.
4) If **only some** attributes are "Default": choose values **only** for those default attributes; keep the user-selected attributes fixed.
5) If **multiple** attributes are "Default": vary those defaults across the 3 combinations so they are **meaningfully different** (no near-duplicates).
6) Favor combinations that are cohesive (palette ↔ pattern ↔ motif ↔ style ↔ finish), print-friendly, and suitable for a premium napkin.
7) Avoid conflicts (e.g., "heavy metallic foil" with a style that demands flat-matte minimalism, or illegible color-on-color).
8) Prioritize contrast (legibility), balanced coverage, and tasteful finish choices.
9) Only use options present in the catalogs for their respective attributes (no new, unseen values).

## Output format (STRICT)
Return **only valid JSON**, no markdown. Use this exact schema:
{{
  "combinations": [
    {{
      "color_palette": "<one from catalog.color_palette>",
      "pattern": "<one from catalog.pattern>",
      "motif": "<one from catalog.motif>",
      "style": "<one from catalog.style>",
      "finish": "<one from catalog.finish>",
      "rationale": "<why this set works; 1-2 short sentences>"
    }},
    {{ ... }},
    {{ ... }}
  ]
}}

- Exactly 3 items in "combinations".
- Each item must fill **all five** attributes.
- Ensure diversity across the 3 items (don’t repeat the same set).

## Notes
- If an attribute in the user selections is **not** "Default", copy it as-is into all 3 combinations.
- If it **is** "Default", you must decide from the catalog (and vary across the 3 suggestions).
"""


class LLMCombiner:
    """
    Builds the prompt for Gemini (or any LLM), calls it via an injected function,
    and returns three validated combinations. Falls back to a local generator if needed.
    """

    def __init__(self, llm_fn: LLMFn):
        """
        llm_fn: a callable taking a prompt string and returning raw text (LLM output).
                Example for Gemini: lambda p: genai_model.generate_content(p).text
        """
        self.llm_fn = llm_fn

    @staticmethod
    def _is_default(val: Any) -> bool:
        return isinstance(val, str) and val.strip().lower() == "default"

    def build_prompt(self, selections: Json, catalog: Json) -> str:
        """
        selections: dict like {"color_palette": "...", "pattern": "...", "motif": "...", "style": "...", "finish": "..."}
        catalog: dict like {"color_palette": [...], "pattern": [...], "motif": [...], "style": [...], "finish": [...]}
        """
        user_selections_json = json.dumps(selections, ensure_ascii=False, indent=2)
        catalog_json = json.dumps(catalog, ensure_ascii=False, indent=2)
        return PROMPT_TEMPLATE.format(
            user_selections_json=user_selections_json,
            catalog_json=catalog_json,
        )

    def _validate_and_fix(self, data: Any, catalog: Json, lock: Json) -> List[Json]:
        """
        Validate JSON from LLM; enforce schema and catalog membership.
        lock: non-default selections that must be kept.
        Returns a list of exactly 3 combinations. Raises on unrecoverable issues.
        """
        if not isinstance(data, dict) or "combinations" not in data:
            raise ValueError("Missing 'combinations' key.")

        combos = data["combinations"]
        if not isinstance(combos, list) or len(combos) != 3:
            raise ValueError("Expected exactly 3 combinations.")

        fixed: List[Json] = []
        for idx, c in enumerate(combos):
            if not isinstance(c, dict):
                raise ValueError(f"Combination {idx} is not an object.")

            out = {}
            for k in ("color_palette", "pattern", "motif", "style", "finish"):
                # enforce locked (non-default) attrs
                if not self._is_default(lock.get(k, "Default")):
                    out[k] = lock[k]
                else:
                    v = c.get(k)
                    if not isinstance(v, str):
                        raise ValueError(f"Combination {idx}: '{k}' must be a string.")
                    if k not in catalog or v not in catalog[k]:
                        raise ValueError(
                            f"Combination {idx}: '{k}' value '{v}' not in catalog."
                        )
                    out[k] = v

            # rationale (optional but nice)
            rationale = c.get("rationale")
            out["rationale"] = rationale if isinstance(rationale, str) else ""

            fixed.append(out)

        # de-duplicate exact duplicates
        unique = []
        seen = set()
        for c in fixed:
            sig = (
                c["color_palette"],
                c["pattern"],
                c["motif"],
                c["style"],
                c["finish"],
            )
            if sig not in seen:
                seen.add(sig)
                unique.append(c)

        # If LLM produced dupes, try to diversify by randomizing remaining catalog choices
        while len(unique) < 3:
            # choose alternatives only for defaulted attributes
            candidate = {}
            for k in ("color_palette", "pattern", "motif", "style", "finish"):
                if not self._is_default(lock.get(k, "Default")):
                    candidate[k] = lock[k]
                else:
                    cand_list = [x for x in catalog.get(k, [])]
                    # avoid picking already used combos
                    if cand_list:
                        candidate[k] = random.choice(cand_list)
                    else:
                        raise ValueError(f"No catalog values for '{k}'.")
            sig = (
                candidate["color_palette"],
                candidate["pattern"],
                candidate["motif"],
                candidate["style"],
                candidate["finish"],
            )
            if sig not in seen:
                candidate["rationale"] = "Auto-diversified fallback."
                seen.add(sig)
                unique.append(candidate)

        return unique[:3]

    def _local_fallback(self, selections: Json, catalog: Json) -> List[Json]:
        """
        Deterministic local generator if LLM fails completely.
        - Respects locked attributes.
        - Varies only default attributes.
        """
        lock = selections
        # build 3 varied picks round-robin
        picks: List[Json] = []
        # prepare indices for default lists
        def_lists: Dict[str, List[str]] = {
            k: (
                catalog.get(k, [])
                if self._is_default(lock.get(k, "Default"))
                else [lock[k]]
            )
            for k in ("color_palette", "pattern", "motif", "style", "finish")
        }
        for i in range(3):
            combo = {}
            for k, vals in def_lists.items():
                if len(vals) == 0:
                    raise ValueError(f"No options for '{k}'.")
                combo[k] = vals[i % len(vals)]
            combo["rationale"] = "Local fallback: balanced rotation across defaults."
            picks.append(combo)
        # ensure uniqueness
        uniq, seen = [], set()
        for c in picks:
            sig = (
                c["color_palette"],
                c["pattern"],
                c["motif"],
                c["style"],
                c["finish"],
            )
            if sig not in seen:
                seen.add(sig)
                uniq.append(c)
        return uniq[:3]

    def generate(self, selections: Json, catalog: Json) -> List[Json]:
        """
        Returns exactly 3 combinations.
        """
        prompt = self.build_prompt(selections, catalog)

        # Try LLM
        try:
            raw = self.llm_fn(prompt)
            # Some LLMs wrap JSON in code fences; strip them safely.
            raw_str = raw.strip()
            if raw_str.startswith("```"):
                raw_str = raw_str.strip("`")
                # remove leading 'json' if present after fence
                if raw_str.startswith("json"):
                    raw_str = raw_str[4:]
            data = json.loads(raw_str)
            combos = self._validate_and_fix(data, catalog, selections)
            return combos
        except Exception:
            # Fallback if anything goes wrong
            return self._local_fallback(selections, catalog)


class GeminiClient:
    def __init__(self):
        pass

    def _get_api_key(
        self,
    ) -> str:
        # Fallback to env var
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "Gemini API key not found. Set Streamlit secret GEMINI_API_KEY "
                "or environment variable GEMINI_API_KEY."
            )
        return key

    def make_gemini_client(self, api_key: Optional[str] = None):
        api_key = api_key or self._get_api_key()
        # google.genai modern client
        return genai.Client(api_key=api_key)

    def gemini_call(self, prompt: str, model: str = "gemini-2.0-flash") -> str:
        """
        Minimal wrapper: (prompt) -> text
        Compatible with current google.genai client methods.
        """
        client = self.make_gemini_client()

        # Try the current/primary method signature first
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
        except Exception:
            # Some SDK versions expose a slightly different surface:
            # fall back to responses.generate(model=..., input=...)
            resp = client.responses.generate(model=model, input=prompt)

        # Extract text robustly
        if getattr(resp, "text", None):
            return resp.text

        # Fallback: stitch candidate parts text if needed
        try:
            cand = resp.candidates[0]
            parts = getattr(cand, "content", getattr(cand, "contents", None)).parts
            chunks = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
            if chunks:
                return "\n".join(chunks)
        except Exception:
            pass

        # Last resort
        return str(resp)
