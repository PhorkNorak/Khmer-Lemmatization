# Khmer Lemmatization Studio

A web-based tool for lemmatizing Khmer text reducing inflected or derived words to their root (base) form.

## Demo

> Paste any Khmer text, click **Lemmatization**, and instantly see each token alongside its root form.

## Features

- **CRF-based tokenization** via [khmer-nltk](https://github.com/VietHoang1512/khmer-nltk) for accurate Khmer word segmentation
- **Dictionary lookup** — maps derived/inflected words to their root forms
- **Token-level results table** — shows every token, its lemma, and whether it changed
- **Summary statistics** — total token count and number of lemmatized tokens
- **Sample text** — one-click load of example Khmer sentences
- **Full dictionary view** — browse all 60 entries directly in the UI

## Project Structure

```
khmer_lemmatization/
├── khmer_lemmatizer_app.py      # Main Gradio application
├── khmer_lemma_dictionary.json  # Lemma dictionary (derived → root)
├── requirements.txt             # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9 or higher

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/khmer-lemmatization.git
cd khmer-lemmatization

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
python khmer_lemmatizer_app.py
```

Then open your browser at `http://127.0.0.1:7860`.

## Dictionary

The lemma dictionary (`khmer_lemma_dictionary.json`) contains **60 entries** mapping derived Khmer words to their root forms. It was compiled from:

> **Teacher Vatha** — [១០០០ពាក្យកម្លាយដោយផ្នត់ដើម (YouTube)](https://youtu.be/mfWl3fV7oMo?si=OuR45gnDqeml2oXw)

Contributions to expand the dictionary are welcome — see [Contributing](#contributing).

## Deployment

### Hugging Face Spaces (Recommended)

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) with **Gradio** as the SDK.
2. Rename `khmer_lemmatizer_app.py` to `app.py`.
3. Push all files to the Space repository:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
cp /path/to/project/* .
git add .
git commit -m "Initial deploy"
git push
```

## Contributing

Contributions are welcome! To add words to the dictionary:

1. Fork the repository
2. Edit `khmer_lemma_dictionary.json` — add entries as `"derived_form": "root_form"`
3. Open a pull request with a brief description

For code contributions, please open an issue first to discuss your proposed change.
