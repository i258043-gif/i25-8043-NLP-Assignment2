import nbformat as nbf, sys
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))
REPO = "https://github.com/i258043-gif/i25-8043-NLP-Assignment2"

md(f"""# DS 5007 - NLP Assignment 2: Neural NLP Pipeline (BBC Urdu)
**Student:** i25-8043  |  **Section:** DS-A  |  **Repository:** {REPO}

Everything is implemented from scratch in PyTorch (no pretrained models, no Gensim, no HuggingFace, no `nn.Transformer` / `nn.MultiheadAttention` / `nn.TransformerEncoder`).

**Important notes**
* The Assignment-1 files were not available, so `cleaned.txt`, `raw.txt` and `Metadata.json` were rebuilt from two public BBC Urdu scrapes (`tools/build_corpus.py`): 504 articles, six BBC sections, 25,273 sentences.
* The assignment lists 11 POS tag names but calls them 12; we added `PART` (particles such as ہی، بھی، نہیں) as the 12th tag.
* The notebook displays results from the full runs stored in `results/`, `figures/`, `embeddings/` and `models/`. Training scripts (`src/part1.py`, `src/part2.py`, `src/part3.py`) regenerate everything (see README). The heavy training cells would take over an hour on a single CPU, so here they are loaded rather than retrained (`RETRAIN = False`).""")

code("""
import sys, os, json, inspect, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch
from IPython.display import Image, display
from src.common import *
RETRAIN = False
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 120)
show = lambda f: display(Image(filename=P("figures", f)))
print("torch", torch.__version__, "| numpy", np.__version__, "| seed", SEED, "| RETRAIN =", RETRAIN)
print("Repository:", "%s")
""" % REPO)

code("""
sents = load_clean_sentences(); raw = load_raw_docs(); meta = load_meta()
toks = [t for s in sents for t in s]
print(f"documents={len(meta)}  sentences={len(sents):,}  tokens={len(toks):,}  distinct tokens={len(set(toks)):,}  raw documents={len(raw)}")
print(pd.Series([m['category_en'] for m in meta]).value_counts().to_string())
print("\\nExample cleaned sentence:", " ".join(sents[100]))
""")

md("## Part 1: Word Embeddings (25 marks)\n### 1.1 TF-IDF weighting")
code("""
r1 = load_json("part1.json")
M = np.load(P("embeddings", "tfidf_matrix.npy"), mmap_mode="r")
print("tfidf_matrix.npy shape (vocabulary x documents):", M.shape, M.dtype, "| vocabulary = 10,000 most frequent tokens + <UNK>")
print("Formula: TF-IDF(w,d) = TF(w,d) * log(N / (1 + df(w))),  N =", r1["tfidf"]["n_docs"])
print("\\nTop-10 most discriminative words per BBC section (mean TF-IDF):")
for c, w in r1["tfidf"]["top"].items(): print(f"  {c:20s}", " ".join(w))
""")

md("### 1.2 PPMI co-occurrence matrix (window k = 5, symmetric)")
code("""
pm = np.load(P("embeddings", "ppmi_matrix.npy"), mmap_mode="r")
nz = sum(int(np.count_nonzero(pm[i:i+500])) for i in range(0, pm.shape[0], 500))
print("ppmi_matrix.npy shape:", pm.shape, "| non-zero entries:", f"{nz:,}", f"({100*nz/pm.shape[0]**2:.2f}% dense)")
print("PPMI(w1,w2) = max(0, log2( P(w1,w2) / (P(w1) P(w2)) ))")
print("\\nTop-5 nearest neighbours by cosine similarity (PPMI vectors, candidates with count >= 20):")
for w, nn in r1["ppmi_nn"].items(): print(f"  {w:10s}", "  ".join(f"{a}({b})" for a, b in nn))
""")
code("""
print("t-SNE of the 200 most frequent tokens, colour-coded by semantic category (counts per category):", r1["tsne_counts"])
show("tsne_ppmi_top200.png")
""")

md("### 2.1 Skip-gram Word2Vec with negative sampling")
code("""
from src import part1
print(inspect.getsource(part1.SkipGram)); print(inspect.getsource(part1.make_pairs))
""")
code("""
from collections import Counter
cnt = Counter(toks); stoi = json.load(open(P("embeddings", "word2idx.json"), encoding="utf8"))
freq = np.array([cnt.get(w, 0) for w in stoi], dtype=float); noise = freq ** 0.75; noise /= noise.sum()
print("Vocabulary size |V| =", len(stoi), "(10,000 + <UNK>)")
print("Noise distribution P(w) ~ f(w)^(3/4): most likely noise words:")
iw = {i: w for w, i in stoi.items()}
for i in np.argsort(-noise)[:8]: print(f"  {iw[i]:8s} unigram={freq[i]/freq.sum():.4f}  noise P={noise[i]:.4f}")
print("\\nHyper-parameters:", r1["config"], "| K = 10 negative samples per positive pair")
print("Skip-gram pairs per epoch ~ 2.32 million (after frequent-word subsampling, t=1e-4)")
""")
code("""
import re
log = open(P("results", "part1_log.txt"), encoding="utf8").read().splitlines()
print("Training loss for C3 (skip-gram on cleaned.txt, d=100), per epoch:")
for l in [l for l in log if re.match(r"\\[C3\\] epoch \\d+/25", l)]: print(" ", l)
h = load_json("p1_C3.json")["hist"]
print("\\nLoss at regular intervals (every 1,000 steps):", [(s, round(v, 3)) for s, v in h if s % 1000 == 0][:12])
show("w2v_loss_curve.png")
E3 = np.load(P("embeddings", "embeddings_w2v.npy")); print("embeddings_w2v.npy = 1/2 (V + U):", E3.shape)
""")

md("### 2.2 Evaluation")
code("""
print("Top-10 nearest neighbours (cosine, C3 embeddings; candidates with corpus count >= 20):\\n")
for k, v in r1["nn10"].items(): print(f"{k:9s}", "  ".join(a for a, _ in v))
""")
code("""
df = pd.DataFrame([{"analogy a : b :: c : ?": f"{t['a']} : {t['b']} :: {t['c']} : ?", "gold": t["gold"], "top-3": " | ".join(t["top3"]),
                    "top-1 correct": t["top1_correct"], "gold in top-3": t["top3_correct"]} for t in r1["analogies"]])
print(df.to_string(index=False))
print(f"\\nCorrect at top-1: {int(df['top-1 correct'].sum())}/10   |   gold within top-3 candidates: {int(df['gold in top-3'].sum())}/10 (requirement: at least 5 correct)")
""")
code("""
print(\"\"\"Assessment (does the embedding space capture meaningful semantics?)
The embeddings capture topical and some morphological relations: the neighbours of عدالت (court) are legal terms (مقدمے, دائر, کورٹ, سماعت, فیصلہ), those of پاکستان are cricket
terms reflecting the news domain, and انڈیا : انڈین follows the nationality pattern. They are noisy, however: with only ~633k training tokens, relational analogies such as
city -> province fail and function words or rare names intrude, so the vectors encode topical association better than precise relations. We kept the 10 analogies fixed
before evaluating and did not tune them to the results.\"\"\")
""")
code("""
rows = []
for k, v in r1["conditions"].items(): rows.append({"Condition": k, "dim": v["dim"], "MRR (20 labelled pairs)": round(v["mrr"], 4), "pairs missing from vocab": v["missing"]})
print(pd.DataFrame(rows).to_string(index=False))
print("\\nTop-5 nearest neighbours of 5 query words per condition:")
for k, v in r1["conditions"].items():
    print("\\n" + k)
    for w, nn in v["nn5"].items(): print(f"   {w:8s}", "  ".join(nn))
c2 = r1["conditions"]["C2 Skip-gram raw (d=100)"]; print("\\nRaw vocabulary entries containing attached punctuation:", c2["raw_vocab_with_punct"], "of", c2["raw_vocab"] - 1)
""")
code("""
print(\"\"\"Discussion of the four conditions
* C1 PPMI obtained the highest MRR (0.223), followed by C2 raw skip-gram (0.132), C4 d=200 (0.114) and C3 cleaned skip-gram (0.113). On this small corpus (~633k tokens) the count-based
  PPMI vectors, which use every co-occurrence exactly, are a stronger baseline than skip-gram trained for 25 epochs; skip-gram with negative sampling normally needs far more data.
* Cleaning did not improve MRR: C2 (raw) is slightly better than C3 (cleaned), although with only 20 pairs the difference is within noise. Qualitatively the five query words give
  similar political/legal neighbours in C2, C3 and C4. Raw text keeps 1,133 punctuation-attached variants in its vocabulary but also keeps information (e.g. sentence-final marks) that cleaning removes.
* Doubling the dimension (C4, d=200) did not help (0.114 vs 0.113): the corpus is the bottleneck, not the capacity of the vectors.
* Overall, PPMI yields the best embeddings by MRR here; among the neural models C2 and C3/C4 are close. C3 is used downstream (Part 2) as required.\"\"\")
""")

md("## Part 2: Sequence Labeling - POS Tagging & NER (25 marks)\n### 3. Dataset preparation")
code("""
from src import lexicons as L
d2 = load_json("part2_data.json")
print("Lexicon entries (hand-crafted, per category):", d2["lexicon_sizes"])
print("Gazetteer entries:", d2["gazetteer_sizes"], "(requirement: >=50 persons, >=50 locations, >=30 organisations; >=200 per major open-class category)")
print("Tag set (12):", L.TAGS); print("NER tags:", L.NER_TAGS)
spl = load_json("part2_data.json")
print("\\nSplit sizes:", spl["sizes"], "| topic distribution:", spl["topic_dist"])
print("Sentences:", sum(spl["sizes"].values()), "| tokens:", spl["n_tokens"], "| entity counts:", spl["entity_counts"])
""")
code("""
split = json.load(open(P("data", "split_annotated.json"), encoding="utf8"))
print("POS label distribution (token counts):"); print(pd.DataFrame(d2["pos_dist"]).fillna(0).astype(int).reindex(L.TAGS).fillna(0).astype(int).to_string())
print("\\nNER label distribution (token counts):"); print(pd.DataFrame(d2["ner_dist"]).fillna(0).astype(int).reindex(L.NER_TAGS).to_string())
print("\\nNote: cleaned.txt contains no punctuation, so PUNC has zero support (it stays in the 12-tag confusion matrix).")
s = next(d for d in split["train"] if any(t != "O" for t in d["ner"]))
print("\\nExample annotated sentence:"); print(pd.DataFrame({"token": s["tokens"], "POS": s["pos"], "NER": s["ner"]}).T.to_string(header=False))
""")
code("""
for f in ("pos_train", "pos_val", "pos_test", "ner_train", "ner_val", "ner_test"):
    n = open(P("data", f + ".conll"), encoding="utf8").read().count("\\n\\n"); print(f"data/{f}.conll : {n} sentences")
print("\\nCaveat: labels are silver (rule-based tagger + gazetteer), not human-checked.")
""")

md("### 4. BiLSTM sequence labeller (2 layers, bidirectional, dropout 0.5, CRF/Viterbi for NER)")
code("""
from src import seqlabel
print(inspect.getsource(seqlabel.CRF)); print(inspect.getsource(seqlabel.BiLSTMTagger))
""")
code("""
sc = seqlabel.crf_selfcheck()
print("CRF verification against brute-force enumeration of all tag paths (2 sequences, one padded):")
for b, v in sc.items(): print(f"  seq {b}: log Z (CRF forward algorithm) = {v['logZ_crf']:.5f} | brute force = {v['logZ_brute']:.5f} | Viterbi path = {v['viterbi']} | brute-force best = {v['brute']}")
print("\\nPadding handling: POS uses cross_entropy(ignore_index=-100) and sequences are packed (pack_padded_sequence); the CRF masks padded steps in both the partition function and the gold score.")
""")
code("""
a2 = load_json("part2_analysis.json"); T = a2["table"]
f = lambda d, k: f"{d[k]['mean']:.3f} +- {d[k]['std']:.3f}"
rows = [{"Task": "POS (macro-F1)", "Mode": m, "Validation F1": f(T[n], "val_f1"), "Test F1": f(T[n], "test_f1")} for m, n in (("Frozen w2v (C3)", "pos_frozen"), ("Fine-tuned w2v (C3)", "pos_ft"))]
rows += [{"Task": "NER (entity F1)", "Mode": m, "Validation F1": f(T[n], "val_f1"), "Test F1": f(T[n], "test_f1")} for m, n in (("Frozen w2v (C3)", "ner_frozen"), ("Fine-tuned w2v (C3)", "ner_ft"))]
print("Frozen vs fine-tuned embeddings (mean +- std over seeds 42, 43):"); print(pd.DataFrame(rows).to_string(index=False))
print("\\nFine-tuning clearly helps (POS +0.25, NER +0.25 validation F1): the frozen C3 vectors were trained on a small corpus and do not encode tag/entity information; the fine-tuned model is used for all later experiments.")
""")
code("""
print("Training setup: Adam (lr 1e-3, weight decay 1e-4), early stopping on validation F1, patience 5 (counted after a 20-epoch warm-up for NER, because the CRF model sits at F1 = 0 for its first epochs).")
print("Best epochs (seed 42):", a2["best_epoch"])
show("pos_ft_loss_curves.png"); show("ner_ft_loss_curves.png")
""")

md("### 5.1 POS tagging evaluation")
code("""
t = T["pos_ft"]; print(f"Test token accuracy: {t['test_acc']['mean']:.3f} +- {t['test_acc']['std']:.3f}   macro-F1 (tags present in test): {t['test_f1']['mean']:.3f} +- {t['test_f1']['std']:.3f}   (mean over 2 seeds; seed 42: acc {a2['pos']['test']['acc']:.3f}, macro-F1 {a2['pos']['test']['macro_f1']:.3f})")
print("\\nFrozen vs fine-tuned summary table (test):")
print(pd.DataFrame([{"Mode": m, "Accuracy": f(T[n], "test_acc"), "Macro-F1": f(T[n], "test_f1")} for m, n in (("Frozen", "pos_frozen"), ("Fine-tuned", "pos_ft"))]).to_string(index=False))
show("pos_confusion_matrix.png")
""")
code("""
print("Three most confused tag pairs (test, seed 42) with example sentences:\\n")
for pair, v in a2["pos"]["top_pairs"].items():
    print(f"== {pair}: {v['count']} confusions")
    for e in v["examples"]: print(f"   token '{e['token']}' gold={e['gold']} predicted={e['pred']} | {e['sentence']}")
print("\\nAnalysis: ADJ/NOUN and NOUN/VERB confusions come from words that genuinely serve both roles in Urdu (participles, nominalised verbs, adjectives used as nouns) and from the")
print("rule-based labels themselves; NOUN/NUM confusions are digit tokens (e.g. years) that map to <UNK> in the embedding vocabulary, so the network cannot see that they are numerals.")
""")

md("### 5.2 NER evaluation (entity-level, conlleval-style)")
code("""
def tab(rep): return pd.DataFrame({k: {"precision": round(v["precision"], 3), "recall": round(v["recall"], 3), "F1": round(v["f1"], 3), "support": v["support"]} for k, v in rep.items()}).T
print("BiLSTM + CRF (seed 42, test) - exact-match entity precision / recall / F1:"); print(tab(a2["ner"]["crf"]).to_string())
print("\\nBiLSTM + softmax (no CRF, seed 42, test):"); print(tab(a2["ner"]["softmax"]).to_string())
print("\\nWith vs without CRF, overall test F1 (mean +- std, 2 seeds):  CRF", f(T["ner_ft"], "test_f1"), "| softmax", f(T["ner_A4_softmax"], "test_f1"))
print("Evaluation uses our own implementation of the conlleval chunk-matching rules (src/seqlabel.py: get_chunks / conlleval); the test set has only 43 entities (15 LOC, 15 MISC, 10 PER, 3 ORG), so scores are noisy.")
""")
code("""
print(f"CRF model errors: {a2['ner']['n_fp_total']} false-positive and {a2['ner']['n_fn_total']} false-negative entities in test+val; first 5 of each:\\n")
print("FALSE POSITIVES (predicted entity not in the gold annotation)")
for i, e in enumerate(a2["ner"]["false_positives"], 1): print(f"{i}. [{e['split']}] '{e['entity']}' predicted {e['predicted']} -> {e['explanation']}\\n   {e['sentence']}")
print("\\nFALSE NEGATIVES (gold entity missed or mis-typed)")
for i, e in enumerate(a2["ner"]["false_negatives"], 1): print(f"{i}. [{e['split']}] '{e['entity']}' gold {e['gold']} -> {e['explanation']}\\n   {e['sentence']}")
""")
code("""
print(\"\"\"Error-analysis summary: the CRF model is conservative - it makes few false positives (8 in test+val) but misses many gazetteer entities (43), especially PER and ORG names, which are
rare in the 350 training sentences and often out-of-vocabulary or unseen as entities during training. Several false positives are plausible entities that the gazetteer simply does not contain,
which reflects the silver-label limitation rather than a model error.\"\"\")
""")

md("### 5.3 Ablation study (same data split; mean +- std over seeds 42, 43)")
code("""
rows = []
for name, label, base in (("A0 (baseline: BiLSTM-2, bidirectional, dropout 0.5, w2v fine-tuned, CRF for NER)", None, None),):
    rows.append({"ID": "Baseline", "Change": "none", "POS val macro-F1": f(T["pos_ft"], "val_f1"), "POS test macro-F1": f(T["pos_ft"], "test_f1"), "POS test acc": f(T["pos_ft"], "test_acc"), "NER val F1": f(T["ner_ft"], "val_f1"), "NER test F1": f(T["ner_ft"], "test_f1")})
for i, (lab, key) in enumerate((("Unidirectional LSTM only", "A1_unidir"), ("No dropout", "A2_nodrop"), ("Random embedding init", "A3_randinit"), ("Softmax instead of CRF (NER)", "A4_softmax")), 1):
    pk = "pos_" + key; nk = "ner_" + key
    rows.append({"ID": f"A{i}", "Change": lab, "POS val macro-F1": f(T[pk], "val_f1") if pk in T else "n/a", "POS test macro-F1": f(T[pk], "test_f1") if pk in T else "n/a",
                 "POS test acc": f(T[pk], "test_acc") if pk in T else "n/a", "NER val F1": f(T[nk], "val_f1"), "NER test F1": f(T[nk], "test_f1")})
print(pd.DataFrame(rows).to_string(index=False))
""")
code("""
print(\"\"\"Discussion of the ablations
A1 (unidirectional): removing the backward LSTM lowers POS test macro-F1 (0.785 vs 0.822) and accuracy (0.912 vs 0.933) and NER test F1 (0.451 vs 0.627, large variance between seeds):
   right-hand context disambiguates tags (e.g. a noun followed by a postposition or verb) and entity boundaries.
A2 (no dropout): POS gets worse (0.798 vs 0.822 test macro-F1; accuracy 0.897 vs 0.933), consistent with over-fitting 350 training sentences. For NER the difference (0.658 vs 0.627) is inside the
   seed noise, so we cannot claim that dropout helps NER here.
A3 (random initialisation): POS is unchanged on test (0.822 vs 0.822; validation is even higher) because the POS labels come from word-level rules, so word identity learned from the training data is
   enough; NER falls from 0.627 to 0.460, so pre-trained Part-1 embeddings do help generalisation to entities with few training examples.
A4 (softmax instead of CRF): NER F1 falls from 0.627 to 0.405: the CRF's learned transition scores forbid invalid sequences (e.g. I-PER after O) and favour consistent multi-token entities.
Caveats: two seeds only, a 76-sentence test set, and silver labels - differences below roughly 0.05 should not be over-interpreted.\"\"\")
""")

md("## Part 3: Transformer Encoder for Topic Classification (20 marks)\n### 6. Dataset preparation")
code("""
d3 = load_json("part3_data.json")
print("Articles:", d3["n"], "| labelled by keyword evidence:", d3["how"].get("keyword evidence"), "| BBC-section fallback (weak evidence, < 3 keyword hits):", d3["how"].get("BBC-section fallback (weak keyword evidence)"))
from src import part3
print("\\nKeyword lists (Urdu translations of the indicative keywords):")
for c, w in part3.KEYWORDS.items(): print(f"  {c:18s} {w}")
print("\\nClass distribution:"); print(pd.DataFrame(d3["dist"]).to_string())
print("\\nSplit sizes:", d3["sizes"], "| each article -> token-ID sequence from cleaned.txt, truncated/padded to", d3["seq_len"], "tokens (", d3["tokens_truncated"], "articles were truncated)")
print("\\nBBC section -> assigned category (counts):"); print(pd.Series(d3["crosstab"]).to_string())
""")

md("### 7. Transformer encoder modules")
code("""
from src import transformer as T3
for obj in (T3.scaled_dot_product_attention, T3.MultiHeadSelfAttention, T3.PositionwiseFFN, T3.SinusoidalPositionalEncoding, T3.EncoderBlock, T3.TransformerClassifier):
    print(inspect.getsource(obj))
""")
code("""
ut = T3.unit_tests()
for k, v in ut.items(): print(f"{k:34s} {v}")
m = T3.TransformerClassifier(10002, 10001, 5); print("\\nArchitecture:", "4 pre-LN blocks, h=4, d_model=128, d_k=d_v=32, d_ff=512, [CLS] token, MLP head 128-64-5; trainable parameters:", f"{sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
""")
code("""
H = {k: load_json(f"p3_{k}.json") for k in ("transformer", "bilstm")}
print("Training: AdamW (lr 5e-4, weight decay 0.01), cosine LR schedule with 50 warm-up steps, 20 epochs, batch 16.\\n")
print(pd.DataFrame({k: v for k, v in H["transformer"]["hist"].items()}).round(3).rename(columns=lambda c: c).assign(epoch=range(1, 21)).set_index("epoch").to_string())
show("transformer_cls_curves.png")
""")

md("### 8. Evaluation")
code("""
e3 = load_json("part3_eval.json")
print(pd.DataFrame({k: {"test accuracy": round(e3[k]["test_acc"], 4), "macro-F1": round(e3[k]["macro_f1"], 4), "best val acc": round(e3[k]["best_val_acc"], 4), "best epoch (val acc)": e3[k]["best_ep"]} for k in ("transformer", "bilstm")}).T.to_string())
print("\\nPer-class F1 (Transformer):", {k: round(v, 3) for k, v in e3["transformer"]["per_class_f1"].items()})
print("Confusion matrix rows = true, cols = predicted, order:", part3.CATS); print(np.array(e3["transformer"]["cm"]))
show("transformer_cls_confusion.png")
""")
code("""
for i, ex in enumerate(e3["attention_examples"], 1):
    print(f"Article {i}: true = predicted = {ex['category']} (confidence {ex['conf']:.2f}); tokens with the highest [CLS] attention (final layer, head average):", ex["cls_top_tokens"][:6])
    show(f"attention_heatmap_{i}.png")
""")
code("""
print("Most [CLS]-attended tokens per class (correctly classified test articles, final layer, mean over heads, tokens occurring >=3 times):")
for c, v in e3["cls_top_tokens_by_class"].items(): print(f"  {c:18s}", "  ".join(f"{a}({b})" for a, b in v[:6]))
print(f"\\nShare of class keywords among each article's top-5 [CLS]-attended tokens: {e3['keyword_fraction_in_top5_cls_attention']:.3f}  (share of keywords among all tokens: {e3['keyword_fraction_baseline']:.3f})")
""")

md("### 8.2 BiLSTM vs Transformer")
code("""
rows = {k: {"test acc": round(e3[k]["test_acc"], 3), "macro-F1": round(e3[k]["macro_f1"], 3), "epoch of min val loss": e3[k]["min_val_loss_epoch"], "epoch of best val acc": e3[k]["best_ep"],
            "epochs to 90% train acc": e3[k]["train_epochs_to_90pct_train_acc"], "sec / epoch (1 CPU)": round(e3[k]["mean_epoch_time"], 1), "final train acc": round(e3[k]["final_train_acc"], 3), "final val acc": round(e3[k]["final_val_acc"], 3)} for k in ("transformer", "bilstm")}
print(pd.DataFrame(rows).T.to_string()); show("cls_val_acc_comparison.png"); show("bilstm_cls_curves.png")
""")
code("""
print(\"\"\"BiLSTM vs Transformer (both trained from the same 352 training articles, 256 tokens)
1. The Transformer achieves the higher accuracy: 68.4% vs 56.6% on the 76 test articles, a gap of 11.8 points (macro-F1 0.673 vs 0.562); with one run and 76 test articles the uncertainty is roughly +-5 points.
2. In terms of fitting the training data the BiLSTM converges in fewer epochs (90% training accuracy after 8 epochs vs 13 for the Transformer, whose learning rate is still warming up for 50 steps).
3. That fast convergence is memorisation: the BiLSTM's validation loss is lowest at epoch 3 and then rises steeply while its training accuracy reaches 100%, whereas the Transformer's validation loss bottoms out at epoch 13.
4. The Transformer was also faster per epoch (18.7 s vs 26.8 s on one CPU), because self-attention processes all 257 positions in parallel while the two bidirectional LSTM layers must step through the sequence sequentially.
5. The attention heatmaps show that the [CLS] query concentrates its weight on a handful of tokens rather than spreading it uniformly.
6. These tokens are often topical: for sports articles the most attended tokens include کرکٹ and سنچری, for international news جنگ (war) receives by far the highest mean weight (0.14), and for economy بجٹ and مہنگائی.
7. Class keywords make up 15.8% of the top-5 [CLS]-attended tokens versus only 2.9% of all tokens, about five times more than chance, although function-like words such as کہا and کام are also attended.
8. Different heads specialise: in the heatmaps one head attends sharply to a few positions while another is more diffuse, giving the encoder several complementary views.
9. Both models over-fit badly (Transformer 98% training vs 61% validation accuracy; BiLSTM 100% vs 43%), which shows how little 352 articles are for models trained from scratch.
10. With only 200-300 articles neither from-scratch network is truly appropriate; our evidence favours the Transformer (better accuracy, smaller train/validation gap, faster) but only if it is kept small and heavily regularised.
11. The BiLSTM's advantage is its sequential inductive bias and fewer position-specific parameters, but its recurrent memory over 256 tokens did not help here.
12. For such a small dataset a TF-IDF + logistic-regression baseline, or a Transformer/BiLSTM initialised with pre-trained embeddings (not allowed in this assignment), would be safer.
13. Caveat: the 5-category labels come from keyword scoring (61 articles fell back to the BBC section), so label noise limits the achievable accuracy for both models.\"\"\")
""")

md("## GitHub submission & file manifest")
code(f"""
print("Public repository (replace with the real URL before submitting):", "{REPO}")
print("Folder structure mirrors the zip layout; see README.md for how to reproduce each part and make_commits.sh for the incremental commit history.\\n")
for root in ("embeddings", "models", "data", "figures", "src"):
    print(root + "/:", ", ".join(sorted(os.listdir(P(root)))[:30]))
""")
nb["cells"] = C; nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "NOTEBOOK.ipynb")
