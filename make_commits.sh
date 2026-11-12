#!/usr/bin/env bash
# Creates an incremental git history (7 commits). Run from the repository root, then:
#   git remote add origin https://github.com/<user>/i25-8043-NLP-Assignment2.git && git branch -M main && git push -u origin main
set -e
git init -q 2>/dev/null || true
c() { git add "$@" 2>/dev/null; }
c .gitignore requirements.txt cleaned.txt raw.txt Metadata.json tools/build_corpus.py src/common.py
git commit -q -m "Add BBC Urdu corpus files, corpus build script and shared utilities"
c src/part1.py embeddings/tfidf_matrix.npy embeddings/embeddings_w2v.npy embeddings/embeddings_w2v_raw.npy embeddings/embeddings_w2v_d200.npy embeddings/word2idx.json embeddings/word2idx_raw.json models/skipgram_c3.pt results/part1.json results/p1_*.json figures/tsne_ppmi_top200.png figures/w2v_*.png
git commit -q -m "Part 1: TF-IDF, PPMI, skip-gram with negative sampling, four-condition evaluation"
c src/lexicons.py src/lexicon_extra.py src/part2_data.py data/pos_*.conll data/ner_*.conll data/split_annotated.json results/part2_data.json
git commit -q -m "Part 2: rule-based POS lexicons, stemmer, NER gazetteers and annotated 500-sentence dataset"
c src/seqlabel.py src/part2.py src/part2_analyze.py models/bilstm_pos.pt models/bilstm_ner.pt results/p2_runs results/part2_analysis.json figures/pos_*.png figures/ner_*.png
git commit -q -m "Part 2: BiLSTM tagger, CRF with Viterbi, ablations A1-A4 and error analysis"
c src/transformer.py src/part3.py src/part3_eval.py data/cls_dataset.npz data/cls_tokens.json models/transformer_cls.pt models/bilstm_cls.pt results/p3_*.json results/part3_*.json figures/attention_heatmap_*.png figures/*cls*.png
git commit -q -m "Part 3: Transformer encoder from scratch, topic classification and BiLSTM comparison"
c *.ipynb tools/build_notebook.py results/part1_*log*.txt
git commit -q -m "Add executed notebook with all results"
c README.md report.pdf tools/build_report.py make_commits.sh
git commit -q -m "Add README with reproduction steps and the final report"
git log --oneline
