from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import gradio as gr
from khmernltk import word_tokenize as khmer_word_tokenize
_GRADIO_IMPORT_ERROR = None

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

SAMPLE_TEXT = ("យើងត្រូវតែផ្គាប់ចិត្តខ្លួនឯងដោយការខ្ជាប់ខ្ជួននូវច្បាប់សីលធម៌ល្អ។ កុំផ្តឹងផ្តល់ ឬផ្តេកផ្តិតទៅលើកំហុសអតីតកាល។ ជីវិតប្រៀបដូចជាការរៀបចំរនាបនិងរនាស់ដើម្បីបណ្ដុះផលល្អ។ ត្រូវកកាយរកនូវចំណេះដឹងថ្មីៗ មិនមែនកកូរចលាចលនោះទេ។ ចេះចចឹករៀនសូត្រពីអ្នកដទៃ មិនមែនសសារឬសសិតរឿងឥតប្រយោជន៍ឡើយ។ ត្រូវជ្រើសរើសដោយឈ្លាសវៃ កុំឲ្យឱកាសល្អៗខ្ចាត់បាត់ ព្រោះយើងខ្ទាស់ចិត្តនឹងសេចក្តីស្អប់ ឬក្បង់យករបស់អាក្រក់។ គំនិតក្លាយជាល្អឬខ្មៅ គឺអាស្រ័យលើយើង ហើយយើងមិនគួរគ្មានមហិច្ឆតាឡើយ។ កុំឆ្កឹះរឿងអ្នកដទៃ កុំធ្វើអ្វីឲ្យឆ្គង។ ត្រូវរៀនសូត្រពីច្បងៗ ធ្វើការងារដោយច្បូតច្បាស់លាស់ ហើយឆ្លាក់ស្នាដៃល្អទុក។ ពេលជួបការលំបាក ត្រូវជ្រុះចោលនូវទុក្ខកង្វល់ ធ្វើចិត្តឲ្យជ្រះថ្លា ឲ្យការយល់ដឹងជ្រាបចូលក្នុងខ្លួន ហើយត្រៀបរៀបចំផែនការដើម្បីត្រងយកតែភាពជោគជ័យ។ ត្រូវផ្ចង់ស្មារតី ទាំងចិត្តផ្កូរឿងល្អ និងផ្ដល់ឲ្យអ្នកដទៃដោយចិត្តបរិសុទ្ធ។ កុំផ្ដាច់ទំនាក់ទំនងល្អ ត្រូវធ្វើអ្វីៗដោយផ្ទាល់ដៃ ហ៊ានផ្ទុកបន្ទុក ហើយប្រឹងប្រែងស្ទាក់ចាប់គោលដៅ ហើយស្រាយបំភ្លឺរាល់បញ្ហាដោយតម្លាភាព។"
)
 
DEFAULT_STATS = {
    "Characters": "0",
    "Total tokens": "0",
    "Lemmatized tokens": "0",
}

EMPTY_TABLE = [["-", "—", "—", "—"]]

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


def render_stats_cards(stats: Dict[str, str]) -> str:
    cards = "".join(
        f"""
        <div class="kpi-card">
            <span class="kpi-label">{label}</span>
            <span class="kpi-value">{value}</span>
        </div>
        """
        for label, value in stats.items()
    )
    return f"""
    <style>
      .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        width: 100%;
      }}
      .kpi-card {{
        border-radius: 16px;
        padding: 18px 20px;
        border: 1px solid var(--border-color-primary, rgba(148, 163, 184, 0.35));
        background: var(--block-background-fill, var(--background-fill-secondary, #fff));
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
      }}
      .kpi-label {{
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--body-text-color-subdued, #64748b);
      }}
      .kpi-value {{
        display: block;
        margin-top: 8px;
        font-size: 2rem;
        font-weight: 700;
        color: var(--body-text-color, #0f172a);
      }}
      @media (prefers-color-scheme: dark) {{
        .kpi-card {{
          box-shadow: none;
        }}
      }}
    </style>
    <div class="kpi-grid">{cards}</div>
    """


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
    return table, render_stats_cards(stats)


def clear_inputs():
    return "", EMPTY_TABLE, render_stats_cards(DEFAULT_STATS)


def build_interface():
    if gr is None:
        raise RuntimeError(
            "Gradio is required to build the UI. Install it via `pip install gradio`."
        ) from _GRADIO_IMPORT_ERROR
    with gr.Blocks(
        title="Khmer Lemmatization Studio",
        theme=gr.themes.Default(),
    ) as demo:
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 1.5rem;">
              <h1>Khmer Lemmatization Studio</h1>
              <p>Paste Khmer sentences, process them instantly, and compare each token to its lemma.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Input text",
                    placeholder="វាយអត្ថបទខ្មែរ ឬបិទភ្ជាប់ពីឯកសារ...",
                    lines=10,
                    autofocus=True,
                )
                with gr.Row():
                    process_btn = gr.Button("Lemmatization", variant="primary")
                    sample_btn = gr.Button("Load Sample")
                    clear_btn = gr.Button("Clear")
                gr.Markdown(
                    """
                    **Workflow**
                    1. Paste or type Khmer text.
                    2. Click "Lemmatization".
                    3. Review tokens, lemmas, and changes on the right.
                    """
                )
            with gr.Column(scale=4):
                stats_panel = gr.HTML(render_stats_cards(DEFAULT_STATS))
                results_table = gr.Dataframe(
                    headers=["#", "Token", "Lemma", "Status"],
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

        # Reference and Dictionary Section
        gr.Markdown("---")
        gr.HTML(
            f"""
            <div style="margin-top: 2rem; padding: 20px; background: var(--background-fill-secondary); border-radius: 12px;">
              <h2 style="margin-top: 0;">Dictionary Source</h2>
              <p style="font-size: 1rem; margin: 12px 0;">
                <strong>Source:</strong> Teacher Vatha -
                <a href="https://youtu.be/mfWl3fV7oMo?si=OuR45gnDqeml2oXw" target="_blank" style="color: var(--link-text-color); text-decoration: underline;">
                  ១០០០ពាក្យកម្លាយដោយផ្នត់ដើម (YouTube)
                </a>
              </p>
              <p style="font-size: 1rem; margin: 12px 0;">
                <strong>Total words our in dictionary:</strong> {len(LEMMA_DICT):,} entries
              </p>
            </div>
            """
        )

        # Display all dictionary entries
        dict_entries = [[i + 1, derived, root] for i, (derived, root) in enumerate(sorted(LEMMA_DICT.items()))]
        gr.Dataframe(
            headers=["#", "ពាក្យកម្លាយ", "ពាក្យឫស"],
            value=dict_entries,
            datatype=["number", "str", "str"],
            interactive=False,
            wrap=True,
            label="Complete Lemmatization Dictionary",
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
