#!/usr/bin/env python3
"""
Modern Gradio interface for Khmer lemmatization powered by khmer-nltk tokenization.
"""
from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - handled dynamically below
    gr = None  # type: ignore[assignment]
    _GRADIO_IMPORT_ERROR: BaseException | None = exc
else:
    _GRADIO_IMPORT_ERROR = None

try:
    from khmernltk import word_tokenize as khmer_word_tokenize
except ImportError as exc:
    raise RuntimeError(
        "khmer-nltk is required for Khmer word segmentation. "
        "Install it via `pip install khmer-nltk`."
    ) from exc

BASE_DIR = Path(__file__).resolve().parent
DICT_PATH = BASE_DIR / "khmer_lemma_dictionary.json"

KHMER_PUNCTUATION = {
    "។",
    "៕",
    "៚",
    "៖",
    "៘",
    "៙",
    "៛",
    "៝",
    "ៗ",
    "៑",
    "៓",
    "។",
    "៘",
    "ៜ",
}
ZERO_WIDTH_CHARS = {"\u200b", "\u200c", "\u200d"}
PUNCTUATION_CHARS = set(string.punctuation) | KHMER_PUNCTUATION

SAMPLE_TEXT = (
    "ខ្ញុំជ្រើសរើសការសិក្សា។ យើងកំពុងរៀន។ គាត់បានធ្វើការងារ។ "
    "ពួកគេបានទៅសាលារៀន។ នេះគឺជាឧទាហរណ៍នៃការវិភាគទម្រង់ដើមពាក្យ។"
)

DEFAULT_STATS = {
    "Characters": "0",
    "Total tokens": "0",
    "Lemmatized tokens": "0",
}

EMPTY_TABLE = [["-", "—", "—", "—"]]

CUSTOM_CSS = """
.gradio-container {
  font-family: 'Inter', 'Noto Sans Khmer', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #f5f7fb;
}

.hero-card {
  background: linear-gradient(135deg, #0f172a, #1d4ed8);
  color: #f8fafc;
  padding: 32px;
  border-radius: 20px;
  box-shadow: 0 20px 35px rgba(15, 23, 42, 0.25);
  margin-bottom: 24px;
}

.hero-card h1 {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 8px;
}

.hero-card p {
  font-size: 1rem;
  color: rgba(248, 250, 252, 0.9);
}

.panel {
  background: #ffffff;
  border-radius: 18px;
  padding: 24px;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
  border: 1px solid #e5e9f2;
}

.button-row button {
  border-radius: 999px !important;
  padding: 12px 20px !important;
  font-weight: 600;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 16px;
  margin-top: 8px;
}

.stat-card {
  background: #f8fafc;
  border-radius: 14px;
  padding: 16px;
  border: 1px solid #e2e8f0;
}

.stat-label {
  font-size: 0.85rem;
  text-transform: uppercase;
  color: #64748b;
  letter-spacing: 0.05em;
}

.stat-value {
  display: block;
  margin-top: 6px;
  font-size: 1.75rem;
  font-weight: 700;
  color: #0f172a;
}

.results-table .table-wrap {
  border-radius: 14px;
  border: 1px solid #e2e8f0;
  overflow: hidden;
}
"""

@dataclass
class LemmaRow:
    token: str
    lemma: str
    changed: bool


def load_dictionary(path: Path = DICT_PATH) -> Dict[str, str]:
    """Load lemma dictionary from JSON and fail loudly if something is wrong."""
    try:
        with path.open(encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Dictionary file not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Dictionary JSON is invalid: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError("Dictionary content must be an object/dict.")
    return data


def _normalize_token(token: str) -> str:
    return "".join(ch for ch in token if ch not in ZERO_WIDTH_CHARS).strip()


def _is_special_token(token: str) -> bool:
    normalized = _normalize_token(token)
    if not normalized:
        return True
    return all(ch in PUNCTUATION_CHARS for ch in normalized)


def tokenize_khmernltk(text: str) -> List[str]:
    """Use the CRF-based word segmenter from khmer-nltk."""
    if not text:
        return []
    tokens = khmer_word_tokenize(text)
    # The library can emit whitespace tokens; filter them out.
    return [tok for tok in tokens if tok.strip()]


def lemmatize_tokens(tokens: Sequence[str], dictionary: Dict[str, str]) -> List[LemmaRow]:
    rows: List[LemmaRow] = []
    for token in tokens:
        lemma = dictionary.get(token, token)
        rows.append(LemmaRow(token=token, lemma=lemma, changed=lemma != token))
    return rows


def format_stats(rows: Sequence[LemmaRow], original_text: str) -> Dict[str, str]:
    char_count = len(original_text)
    token_count = len(rows)
    changed = sum(1 for row in rows if row.changed)
    return {
        "Characters": f"{char_count:,}",
        "Total tokens": f"{token_count:,}",
        "Lemmatized tokens": f"{changed:,}",
    }


def render_stats_html(stats: Dict[str, str]) -> str:
    cards = "".join(
        f"""
        <div class="stat-card">
            <span class="stat-label">{label}</span>
            <span class="stat-value">{value}</span>
        </div>
        """
        for label, value in stats.items()
    )
    return f'<div class="stats-grid">{cards}</div>'


def run_pipeline(text: str) -> tuple:
    tokens = [
        token
        for token in tokenize_khmernltk(text or "")
        if not _is_special_token(token)
    ]
    rows = lemmatize_tokens(tokens, LEMMA_DICT)
    table = [[idx + 1, row.token, row.lemma, "Yes" if row.changed else "No"] for idx, row in enumerate(rows)]
    stats = format_stats(rows, text or "")

    if not table:
        # Inform the UI that there is nothing to display.
        table = [["-", "—", "—", "—"]]
    return table, render_stats_html(stats)


def clear_inputs():
    return "", EMPTY_TABLE, render_stats_html(DEFAULT_STATS)


def build_interface():
    if gr is None:
        raise RuntimeError(
            "Gradio is required to build the UI. Install it via `pip install gradio`."
        ) from _GRADIO_IMPORT_ERROR
    with gr.Blocks(
        title="Khmer Lemmatization Studio",
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="gray"),
        css=CUSTOM_CSS,
    ) as demo:
        gr.HTML(
            """
            <div class="hero-card">
              <h1>Khmer Lemmatization Studio</h1>
              <p>Paste Khmer sentences, process them instantly, and compare each token to its lemma with a polished review surface.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=3, elem_classes="panel"):
                text_input = gr.Textbox(
                    label="Input text",
                    placeholder="វាយអត្ថបទខ្មែរ ឬបិទភ្ជាប់ពីឯកសារ...",
                    lines=10,
                    autofocus=True,
                )
                with gr.Row(elem_classes="button-row"):
                    process_btn = gr.Button("Run Lemmatization", variant="primary")
                    sample_btn = gr.Button("Load Sample")
                    clear_btn = gr.Button("Clear")
                gr.Markdown(
                    """
                    **Workflow**
                    1. Paste or type Khmer text.
                    2. Click "Run Lemmatization".
                    3. Review tokens, lemmas, and changes on the right.
                    """
                )
            with gr.Column(scale=4, elem_classes="panel"):
                stats_panel = gr.HTML(render_stats_html(DEFAULT_STATS))
                results_table = gr.Dataframe(
                    headers=["#", "Token", "Lemma", "Changed?"],
                    datatype=["number", "str", "str", "str"],
                    value=EMPTY_TABLE,
                    interactive=False,
                    wrap=True,
                    label="Results",
                )

        process_btn.click(fn=run_pipeline, inputs=text_input, outputs=[results_table, stats_panel])
        sample_btn.click(
            fn=lambda: (SAMPLE_TEXT, *run_pipeline(SAMPLE_TEXT)),
            inputs=None,
            outputs=[text_input, results_table, stats_panel],
        )
        clear_btn.click(
            fn=clear_inputs,
            inputs=None,
            outputs=[text_input, results_table, stats_panel],
        )
    return demo


LEMMA_DICT = load_dictionary()


if __name__ == "__main__":
    if gr is None:
        raise RuntimeError(
            "Gradio is required to launch the UI. Install it via `pip install gradio`."
        ) from _GRADIO_IMPORT_ERROR
    app = build_interface()
    app.launch()
