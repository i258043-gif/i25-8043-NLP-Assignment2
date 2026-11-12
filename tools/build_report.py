from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib import colors
import sys
REPO = "https://github.com/i258043-gif/i25-8043-NLP-Assignment2"
body = ParagraphStyle("b", fontName="Times-Roman", fontSize=12, leading=18, alignment=TA_JUSTIFY, spaceAfter=0)
head = ParagraphStyle("h", parent=body, fontName="Times-Bold", spaceBefore=2)
title = ParagraphStyle("t", parent=body, fontName="Times-Bold", alignment=TA_CENTER)
cell = ParagraphStyle("c", fontName="Times-Roman", fontSize=12, leading=14)
cellb = ParagraphStyle("cb", parent=cell, fontName="Times-Bold")
M = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
doc = SimpleDocTemplate("report.pdf", pagesize=letter, leftMargin=M*inch, rightMargin=M*inch, topMargin=M*inch, bottomMargin=M*inch,
                        title="DS 5007 Assignment 2 Report", author="i25-8043")
S = []
def P(t, st=body): S.append(Paragraph(t, st))
def T(rows, widths):
    data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w*inch for w in widths]); t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.4, colors.black), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e6e6e6")),
        ("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING", (0,0), (-1,-1), 1), ("BOTTOMPADDING", (0,0), (-1,-1), 1)])); S.append(t); S.append(Spacer(1, 4))

P("DS 5007 Assignment 2: Neural NLP Pipeline on BBC Urdu (i25-8043, DS-A)", title)
P("1. Overview", head)
P(f"All components were implemented from scratch in PyTorch (no pretrained models, Gensim, HuggingFace or <i>nn.Transformer</i>). Code, notebook and README are in the public repository {REPO}. "
  "The Assignment 1 files were unavailable, so <i>cleaned.txt</i>, <i>raw.txt</i> and <i>Metadata.json</i> were rebuilt from two public BBC Urdu scrapes: 504 articles in six sections, 25,273 sentences and about 633k tokens. "
  "The brief names 11 POS tags but calls them 12, so a PART tag was added. Results use two seeds (Part 2) or one run (Parts 1 and 3) on small test sets and should be read with that uncertainty.")
P("2. Part 1 Results: Word Embeddings", head)
P("<b>TF-IDF and PPMI.</b> The vocabulary is the 10,000 most frequent tokens plus &lt;UNK&gt;. TF-IDF (TF x log(N/(1+df))) and a symmetric k=5 PPMI matrix were saved, with top-10 discriminative words per section and a t-SNE map of the 200 most frequent tokens. "
  "<b>Skip-gram.</b> Separate centre and context matrices (V, U), noise distribution f(w)<super>3/4</super> with K=10, binary cross-entropy, d=100, Adam at 0.001, batch 1,024 and 25 epochs; the loss fell from 3.57 to 2.45 and the average (V+U)/2 was saved. "
  "Neighbours are sensible (the court word <i>adalat</i> gives case, filed, hearing, verdict), and 5 of 10 fixed analogies have the gold answer in the top 3 (2 at top-1; for example India : Indian). Relational analogies such as city : province fail, so the space captures topical association more than precise relations.")
T([["Condition", "d", "MRR (20 pairs)"], ["C1 PPMI baseline", "10,001", "0.223"], ["C2 Skip-gram, raw.txt", "100", "0.132"], ["C3 Skip-gram, cleaned.txt", "100", "0.113"], ["C4 Skip-gram, cleaned.txt", "200", "0.114"]], [2.8, 1.0, 1.5])
P("PPMI is best on this small corpus; cleaning did not help (C2 is slightly above C3, within noise for 20 pairs) and doubling d gave no gain, so data, not capacity, is the bottleneck. Candidates were limited to words seen at least 20 times for every condition.")
P("3. Part 2 Results: POS Tagging and NER", head)
P("500 sentences (200 Pakistan, 150 Sports, 150 Entertainment) were tagged by a rule-based tagger (stemmer plus lexicons of 845 nouns, 217 verbs, 450 adjectives, 206 adverbs) and a gazetteer (147 persons, 113 locations, 98 organisations), then split 70/15/15 by topic (350/74/76). "
  "These silver labels are not human-checked. The 2-layer BiLSTM (dropout 0.5, C3 initialisation) uses a linear softmax for POS and a from-scratch CRF with Viterbi decoding for NER (verified against brute force). "
  "Fine-tuning the embeddings beat freezing them (validation F1 0.825 vs 0.576 for POS, 0.745 vs 0.493 for NER). On the test set POS accuracy is 0.933 and macro-F1 0.822; the most confused pairs are ADJ/NOUN (29), NOUN/VERB (28) and NOUN/NUM (20), the last caused by digits mapping to &lt;UNK&gt;. "
  "NER entity-level F1 is 0.627 with the CRF (LOC 0.80, MISC 0.64, PER 0.31, ORG 0.00 on only 3 entities) against 0.405 without it. The CRF is conservative (8 false positives, 43 misses): errors are mostly names unseen in training and fragments of abbreviations.")
T([["Setting (mean of 2 seeds)", "POS test macro-F1", "NER test F1"], ["Baseline (BiLSTM, fine-tuned, CRF)", "0.822", "0.627"], ["Frozen embeddings", "0.563", "0.394"], ["A1 unidirectional", "0.785", "0.451"],
   ["A2 no dropout", "0.798", "0.658"], ["A3 random initialisation", "0.822", "0.460"], ["A4 softmax instead of CRF", "n/a", "0.405"]], [2.9, 1.5, 1.3])
P("Backward context and the CRF matter most; pretrained embeddings help NER but not POS, whose rule-derived labels depend only on word identity; dropout helps POS, and its NER effect is within seed noise.")
P("4. Part 3 Results: Transformer Topic Classification", head)
P("Articles were assigned to five categories by Urdu keyword scoring (61 of 504 fell back to the BBC section) and split 352/76/76. The encoder (4 pre-LN blocks, 4 heads, d_model 128, [CLS] head 128-64-5, sinusoidal buffer, separate per-head projections, 2.08M parameters including embeddings; masking and padding invariance unit-tested) was trained with AdamW and cosine warm-up for 20 epochs. "
  "It reaches 68.4% test accuracy and 0.673 macro-F1, versus 56.6% and 0.562 for a BiLSTM classifier. The Transformer was also faster per epoch (18.7 s vs 26.8 s) and its validation loss bottomed at epoch 13 (BiLSTM: epoch 3, then over-fitting). "
  "Final-layer [CLS] attention favours topical words (cricket, war, budget, inflation): class keywords are 15.8% of the top-5 attended tokens against 2.9% of all tokens. Both models over-fit (training accuracy 98-100%), so for 200-300 articles a small regularised model or a TF-IDF baseline is safer.")
T([["Model", "Test acc.", "Macro-F1", "Best val epoch", "s/epoch"], ["Transformer", "0.684", "0.673", "17", "18.7"], ["BiLSTM", "0.566", "0.562", "14", "26.8"]], [1.9, 1.1, 1.1, 1.4, 1.0])
P("5. Conclusion", head)
P("The full pipeline was built and evaluated. Count-based PPMI beat skip-gram on a 633k-token corpus, fine-tuning and the CRF were the most useful choices for sequence labelling, and the Transformer outperformed the BiLSTM on topic classification. "
  "Main limitations are the substitute corpus, silver POS and NER labels, keyword-derived topic labels, few seeds and very small test sets.")
doc.build(S)
