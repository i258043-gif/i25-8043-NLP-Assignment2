import sys
sys.path.insert(0, __file__.rsplit("/src/", 1)[0])
from src.seqlabel import *
from src.part2 import CFGS, SEEDS
from src import lexicons as L

R = {(n, s): json.load(open(P("results", "p2_runs", f"{n}_{s}.json"), encoding="utf8")) for n in CFGS for s in SEEDS}
split = load_split()


def ms(vals): return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
def key(n): return "macro_f1" if n.startswith("pos") else "f1"
out = {}

# ---- frozen vs fine-tuned and ablation tables
tab = {}
for n in CFGS:
    t = "pos" if n.startswith("pos") else "ner"
    row = dict(val_f1=[R[n, s]["val"][key(n)] for s in SEEDS], test_f1=[R[n, s]["test"][key(n)] for s in SEEDS],
               epochs=[R[n, s]["best_epoch"] for s in SEEDS])
    if t == "pos": row["test_acc"] = [R[n, s]["test"]["acc"] for s in SEEDS]
    tab[n] = {k: dict(mean=float(np.mean(v)), std=float(np.std(v)), runs=v) for k, v in row.items()}
out["table"] = tab
for n, r in tab.items():
    print(n, {k: f"{v['mean']:.3f}±{v['std']:.3f}" for k, v in r.items()})

# ---- POS confusion matrix (seed 42 fine-tuned, test)
r = R["pos_ft", 42]; g = [x for s in r["gold"] for x in s]; p = [x for s in r["pred"] for x in s]
cm = np.zeros((12, 12), int); ti = {t: i for i, t in enumerate(TAGS)}
for a, b in zip(g, p): cm[ti[a], ti[b]] += 1
fig, ax = plt.subplots(figsize=(8.5, 7)); im = ax.imshow(np.log1p(cm), cmap="Blues")
ax.set_xticks(range(12)); ax.set_yticks(range(12)); ax.set_xticklabels(TAGS, rotation=45); ax.set_yticklabels(TAGS)
for i in range(12):
    for j in range(12):
        if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if np.log1p(cm[i, j]) > 3.5 else "black")
ax.set_xlabel("Predicted tag"); ax.set_ylabel("Gold tag"); ax.set_title("POS confusion matrix on the test set (BiLSTM, fine-tuned w2v; colour = log counts)")
fig.colorbar(im, label="log(1 + count)"); fig.savefig(P("figures", "pos_confusion_matrix.png")); plt.close(fig)
pairs = Counter()
for i in range(12):
    for j in range(12):
        if i != j and cm[i, j]: pairs[tuple(sorted((TAGS[i], TAGS[j])))] += cm[i, j]
top = [pr for pr, c in pairs.most_common(3)]
ex = {}
for a, b in top:
    exs = []
    for si, (d, gs, ps) in enumerate(zip(split["test"], r["gold"], r["pred"])):
        for ti_, (w, x, y) in enumerate(zip(d["tokens"], gs, ps)):
            if {x, y} == {a, b} and x != y: exs.append(dict(sentence=" ".join(d["tokens"]), token=w, gold=x, pred=y, pos=ti_))
    ex[f"{a}<->{b}"] = dict(count=int(pairs[(a, b)]), examples=exs[:2], n_found=len(exs))
    print(a, b, pairs[(a, b)], [(e["token"], e["gold"], e["pred"]) for e in exs[:2]])
out["pos"] = dict(confusion=cm.tolist(), top_pairs=ex, test=R["pos_ft", 42]["test"],
                  per_tag_support={t: int(cm[i].sum()) for t, i in ti.items()})

# ---- NER: per-type results (seed 42), with/without CRF
def ner_report(name):
    return conlleval(R[name, 42]["gold"], R[name, 42]["pred"])
out["ner"] = dict(crf=ner_report("ner_ft"), softmax=ner_report("ner_A4_softmax"), frozen=ner_report("ner_frozen"))
for k in ("crf", "softmax"):
    print(k, {t: f"P{v['precision']:.2f} R{v['recall']:.2f} F{v['f1']:.2f} n{v['support']}" for t, v in out["ner"][k].items()})

# ---- NER error analysis (CRF model, seed 42; test first, then val to reach 5)
train_ent = Counter()
for d in split["train"]:
    for (t, s, e) in get_chunks(d["ner"]): train_ent[(t, tuple(d["tokens"][s:e + 1]))] += 1
stoi = json.load(open(P("embeddings", "word2idx.json"), encoding="utf8"))
fps, fns = [], []
for part, items, G, Pp in (("test", split["test"], R["ner_ft", 42]["gold"], R["ner_ft", 42]["pred"]),
                           ("val", split["val"], R["ner_ft", 42]["vgold"], R["ner_ft", 42]["vpred"])):
    for d, g_, p_ in zip(items, G, Pp):
        GC, PC = get_chunks(g_), get_chunks(p_)
        for c in PC - GC:
            ents = " ".join(d["tokens"][c[1]:c[2] + 1]); ov = [x for x in GC if not (x[2] < c[1] or x[1] > c[2])]
            if ov:
                o = ov[0]; why = (f"Type confusion: predicted {c[0]} but the gold label is {o[0]}" if (o[1], o[2]) == (c[1], c[2]) and o[0] != c[0]
                                  else f"Boundary error: predicted span {c[1]}-{c[2]} overlaps gold {o[0]} span {o[1]}-{o[2]}")
            else:
                oov = [w for w in d["tokens"][c[1]:c[2] + 1] if stoi.get(w, 0) == 0]
                why = ("Spurious entity: the gazetteer-derived gold marks these tokens O; the model generalised a name-like context/pattern" +
                       (" (contains out-of-vocabulary tokens)" if oov else "") + (" - possibly a genuine entity missing from the gazetteer" if not any(w in L.GAZ_WORDS for w in d["tokens"][c[1]:c[2] + 1]) else ""))
            fps.append(dict(split=part, entity=ents, predicted=c[0], explanation=why, sentence=" ".join(d["tokens"])))
        for c in GC - PC:
            ents = " ".join(d["tokens"][c[1]:c[2] + 1]); ov = [x for x in PC if not (x[2] < c[1] or x[1] > c[2])]
            seen = train_ent[(c[0], tuple(d["tokens"][c[1]:c[2] + 1]))]
            if ov: why = f"Partially/incorrectly recognised: predicted {ov[0][0]} span {ov[0][1]}-{ov[0][2]} versus gold {c[0]} span {c[1]}-{c[2]}"
            else:
                oov = [w for w in d["tokens"][c[1]:c[2] + 1] if stoi.get(w, 0) == 0]
                why = ("Missed entity (predicted O): " + ("this entity string never occurs as an entity in the training split" if seen == 0 else f"seen only {seen}x in training") +
                       ("; tokens are out-of-vocabulary for the embeddings" if oov else ""))
            fns.append(dict(split=part, entity=ents, gold=c[0], explanation=why, sentence=" ".join(d["tokens"])))
out["ner"]["false_positives"] = fps[:5]; out["ner"]["false_negatives"] = fns[:5]
out["ner"]["n_fp_total"] = len(fps); out["ner"]["n_fn_total"] = len(fns)
print("FP", len(fps), "FN", len(fns))

# ---- curves
for name, title in (("pos_ft", "POS"), ("ner_ft", "NER (CRF)")):
    h = R[name, 42]["hist"]; ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(ep, h["train_loss"], label="Training loss"); ax.plot(ep, h["val_loss"], label="Validation loss")
    ax.axvline(R[name, 42]["best_epoch"], ls="--", c="gray", label=f"Best val F1 (epoch {R[name, 42]['best_epoch']})")
    ax.set_title(f"BiLSTM {title}: training and validation loss per epoch"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss" if name == "pos_ft" else "CRF negative log-likelihood")
    ax.legend(); ax.grid(alpha=.3); fig.savefig(P("figures", f"{name}_loss_curves.png")); plt.close(fig)
out["curves"] = {n: R[n, 42]["hist"] for n in ("pos_ft", "ner_ft")}; out["best_epoch"] = {n: R[n, 42]["best_epoch"] for n in ("pos_ft", "ner_ft")}
out["crf_selfcheck"] = crf_selfcheck()
save_json(out, "part2_analysis.json")
