# i25-8043-NLP-Assignment2 — Neural NLP Pipeline on BBC Urdu (DS 5007)

**Student ID:** i25-8043  
**Section:** DS-A  
**Repository:** https://github.com/i258043-gif/i25-8043-NLP-Assignment2

This project implements a three-part Urdu NLP pipeline in PyTorch:
1. TF-IDF, PPMI, and Skip-gram word representations.
2. BiLSTM-based POS tagging and CRF-based NER.
3. A Transformer encoder built from scratch for news-topic classification, compared with a BiLSTM baseline.

The implementation does not use pretrained models, Gensim, Hugging Face, or PyTorch's `nn.Transformer` / `nn.MultiheadAttention` / `nn.TransformerEncoder`.

## Project structure

```text
i25-8043_Assignment2_DS-A/
├── i25-8043_Assignment2_DS-A.ipynb   # Executed notebook
├── report.pdf                       # Final six-page assignment report
├── README.md
├── GITHUB_UPLOAD_GUIDE.md
├── requirements.txt
├── make_commits.sh
├── src/                             # Part 1, Part 2, Part 3 implementation
├── tools/                           # Corpus, notebook, and report builders
├── data/                            # POS/NER splits and topic-classification data
├── embeddings/                      # TF-IDF and Skip-gram matrices/vectors
├── models/                          # Trained model checkpoints
├── results/                         # Metrics, logs, and ablation outputs
├── figures/                          # Plots and attention visualisations
├── cleaned.txt
├── raw.txt
└── Metadata.json
```

## Setup

Python 3.10 or later is recommended. Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

Run commands from this directory.

## Reproducing the experiments

**Corpus:** `cleaned.txt`, `raw.txt`, and `Metadata.json` are included. They were reconstructed from two public BBC Urdu scrapes because the original Assignment 1 corpus files were unavailable.

**Part 1 — word representations**
```bash
python -m src.part1 A
python -m src.part1 C3
python -m src.part1 C2
python -m src.part1 C4
python -m src.part1 eval
```

**Part 2 — POS tagging and NER**
```bash
python -m src.part2_data
python -m src.part2 run
python -m src.part2_analyze
```

**Part 3 — topic classification**
```bash
python -m src.part3 data
python -m src.part3 transformer
python -m src.part3 bilstm
python -m src.part3_eval
```

The included notebook displays the stored experiment outputs. Its training configuration is set to load the saved results rather than automatically retrain all models. See `GITHUB_UPLOAD_GUIDE.md` for repository creation and upload instructions.

## Important notes

- The original Assignment 1 files were unavailable, so the corpus was reconstructed from two public BBC Urdu scrapes.
- POS and NER labels are silver labels produced by rules and gazetteers; they are not manually verified gold labels.
- Topic labels were created by keyword scoring, with BBC section fallback for 61 articles.
- The POS tag set contains 12 tags; `PART` was added because the assignment brief names 11 tags while referring to 12.
- Part 2 results use two seeds (42 and 43); Parts 1 and 3 report one run. The report discusses small test sets and overfitting.
- `embeddings/ppmi_matrix.npy` is approximately 400 MB and is excluded from Git by `.gitignore` to avoid GitHub's 100 MB per-file limit. Regenerate it with `python -m src.part1 A`.
- The final report is the `report.pdf` included here. `tools/build_report.py` is a compact report-generation utility and may produce a different abbreviated report if run again.
