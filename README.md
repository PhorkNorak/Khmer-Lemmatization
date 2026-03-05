# Khmer Lemmatization Studio

A web-based tool for lemmatizing Khmer text reducing inflected or derived words to their root (base) form.

## Demo

![Demo](demo.gif)

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
├── app.py                       # Main Gradio application
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
git clone https://github.com/PhorkNorak/Khmer-Lemmatization.git
cd Khmer-Lemmatization

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
```

Then open your browser at `http://127.0.0.1:7860`.

## Dictionary

The lemma dictionary (`khmer_lemma_dictionary.json`) contains **60 entries** mapping derived Khmer words to their root forms. It was compiled from:

> **Teacher Vatha** — [១០០០ពាក្យកម្លាយដោយផ្នត់ដើម (YouTube)](https://youtu.be/mfWl3fV7oMo?si=OuR45gnDqeml2oXw)

Contributions to expand the dictionary are welcome — see [Contributing](#contributing).

## Deployment

This app is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/Norak007/Khmer-Lemmatization-Studio).

## Contributing

Contributions are welcome! To add words to the dictionary:

1. Fork the repository
2. Edit `khmer_lemma_dictionary.json` — add entries as `"derived_form": "root_form"`
3. Open a pull request with a brief description

For code contributions, please open an issue first to discuss your proposed change.


## Developers

| Name | Role | Portfolio |
|------|------|-----------|
| Phork Norak | Full-Stack Developer | [phorknorak.vercel.app](https://phorknorak.vercel.app/) |
| Nhor Povketya | Data Collection | [povketya.github.io/ketyanhor](https://povketya.github.io/ketyanhor/) |
| Ly Hor | Data Collection | [final-portfolio-kappa-rust.vercel.app](https://final-portfolio-kappa-rust.vercel.app/) |

---
