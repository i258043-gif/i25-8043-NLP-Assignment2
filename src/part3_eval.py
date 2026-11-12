import sys
sys.path.insert(0, __file__.rsplit("/src/", 1)[0])
from src.part3 import *
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

ids, y, tr, va, te = load_ds(); info = load_json("part3_data.json"); pad, V = info["pad_id"], info["vocab"]
X = torch.tensor(ids); toks = json.load(open(P("data", "cls_tokens.json"), encoding="utf8")); meta = load_meta()
tf = TransformerClassifier(V + 1, pad, 5); ck = torch.load(P("models", "transformer_cls.pt"), weights_only=False); tf.load_state_dict(ck["state"]); tf.eval()
bl = BiLSTMClassifier(np.load(P("embeddings", "embeddings_w2v.npy")), pad, 5); ckb = torch.load(P("models", "bilstm_cls.pt"), weights_only=False); bl.load_state_dict(ckb["state"]); bl.eval()
H = {k: load_json(f"p3_{k}.json") for k in ("transformer", "bilstm")}
out = dict(best_epoch={k: H[k]["best_ep"] for k in H})
with torch.no_grad():
    pt = tf(X[te]).argmax(1).numpy(); pb = bl(X[te]).argmax(1).numpy()
yt = y[te]
for name, p in (("transformer", pt), ("bilstm", pb)):
    cm = confusion_matrix(yt, p, labels=range(5))
    out[name] = dict(test_acc=float(accuracy_score(yt, p)), macro_f1=float(f1_score(yt, p, average="macro")), cm=cm.tolist(),
                     per_class_f1={CATS[i]: float(v) for i, v in enumerate(f1_score(yt, p, average=None, labels=range(5)))},
                     best_val_acc=H[name]["best_val"], best_ep=H[name]["best_ep"], mean_epoch_time=float(np.mean(H[name]["hist"]["epoch_time"])),
                     final_train_acc=H[name]["hist"]["train_acc"][-1], final_val_acc=H[name]["hist"]["val_acc"][-1],
                     epochs_to_95pct_of_best_val=next(i + 1 for i, a in enumerate(H[name]["hist"]["val_acc"]) if a >= 0.95 * H[name]["best_val"]),
                     min_val_loss_epoch=int(np.argmin(H[name]["hist"]["val_loss"])) + 1,
                     train_epochs_to_90pct_train_acc=next((i + 1 for i, a in enumerate(H[name]["hist"]["train_acc"]) if a >= 0.9), None))
    fig, ax = plt.subplots(figsize=(6.2, 5)); im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(CATS, rotation=30, ha="right"); ax.set_yticklabels(CATS)
    for i in range(5):
        for j in range(5): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted category"); ax.set_ylabel("True category"); ax.set_title(f"{name.capitalize()} test confusion matrix (5 x 5)"); fig.colorbar(im)
    fig.savefig(P("figures", f"{name}_cls_confusion.png")); plt.close(fig)
    print(name, {k: v for k, v in out[name].items() if k not in ("cm",)})

# training curves (loss and accuracy per epoch)
for name in ("transformer", "bilstm"):
    h = H[name]["hist"]; ep = range(1, 21); fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, h["train_loss"], label="Training"); ax[0].plot(ep, h["val_loss"], label="Validation"); ax[0].set_title(f"{name.capitalize()} classifier: loss per epoch"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Cross-entropy loss")
    ax[1].plot(ep, h["train_acc"], label="Training"); ax[1].plot(ep, h["val_acc"], label="Validation"); ax[1].set_title(f"{name.capitalize()} classifier: accuracy per epoch"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy")
    for a in ax: a.legend(); a.grid(alpha=.3)
    fig.savefig(P("figures", f"{name}_cls_curves.png")); plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 4))
for name in ("transformer", "bilstm"): ax.plot(range(1, 21), H[name]["hist"]["val_acc"], label=name.capitalize())
ax.set_title("Validation accuracy per epoch: Transformer vs BiLSTM"); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy"); ax.legend(); ax.grid(alpha=.3)
fig.savefig(P("figures", "cls_val_acc_comparison.png")); plt.close(fig)

# attention: 3 correctly classified test articles (different classes), final layer, heads 0 and 1
with torch.no_grad(): lg, attns = tf(X[te], return_attn=True)
prob = torch.softmax(lg, 1); correct = [i for i in range(len(te)) if pt[i] == yt[i]]
chosen = []
for c in range(5):
    cand = sorted([i for i in correct if yt[i] == c], key=lambda i: -float(prob[i, c]))
    if cand and len(chosen) < 3 and c in (1, 3, 4): chosen.append(cand[0])
last = attns[-1]                                                     # [B, heads, 257, 257]
NSHOW = 21
exs = []
for k, i in enumerate(chosen):
    t = toks[te[i]]; labs = ["[CLS]"] + t[:NSHOW - 1]
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.2))
    for hd, ax in zip((0, 1), axs):
        M = last[i, hd, :NSHOW, :NSHOW].numpy(); im = ax.imshow(M, cmap="viridis")
        ax.set_xticks(range(NSHOW)); ax.set_yticks(range(NSHOW)); lab = [ur(l) if l != "[CLS]" else "[CLS]" for l in labs]
        ax.set_xticklabels(lab, rotation=90, fontsize=11); ax.set_yticklabels(lab, fontsize=11)
        ax.set_title(f"Head {hd+1} (final layer) - {CATS[yt[i]]} article"); ax.set_xlabel("Key token (attended to)"); ax.set_ylabel("Query token"); fig.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle(f"Attention weights, test article {k+1}: true = predicted = {CATS[yt[i]]} (first {NSHOW-1} tokens)"); fig.savefig(P("figures", f"attention_heatmap_{k+1}.png")); plt.close(fig)
    # which tokens does [CLS] attend to (all heads averaged, full article)?
    w = last[i, :, 0, 1:1 + len(t)].mean(0).numpy(); top = np.argsort(-w)[:8]
    exs.append(dict(article=int(te[i]), title=meta[int(te[i])]["title"], category=CATS[yt[i]], conf=float(prob[i, yt[i]]), cls_top_tokens=[(t[j], round(float(w[j]), 4)) for j in top]))
out["attention_examples"] = exs
# aggregate: most CLS-attended tokens per class (correctly classified test articles, last layer, head-average), tokens with >=3 occurrences
agg = {c: Counter() for c in range(5)}; cnt = {c: Counter() for c in range(5)}
for i in correct:
    t = toks[te[i]]; w = last[i, :, 0, 1:1 + len(t)].mean(0).numpy()
    for tok, wi in zip(t, w): agg[yt[i]][tok] += float(wi); cnt[yt[i]][tok] += 1
kw = {c: set(KEYWORDS[CATS[c]].split()) for c in range(5)}
out["cls_top_tokens_by_class"] = {CATS[c]: [(tok, round(agg[c][tok] / cnt[c][tok], 4)) for tok in sorted([k for k in agg[c] if cnt[c][k] >= 3], key=lambda k: -agg[c][k] / cnt[c][k])[:8]] for c in range(5)}
fr = []
for i in correct:
    t = toks[te[i]]; w = last[i, :, 0, 1:1 + len(t)].mean(0).numpy(); top5 = [t[j] for j in np.argsort(-w)[:5]]
    fr.append(np.mean([tok in kw[yt[i]] for tok in top5]))
base = np.mean([np.mean([tok in kw[yt[i]] for tok in toks[te[i]]]) for i in correct])
out["keyword_fraction_in_top5_cls_attention"] = float(np.mean(fr)); out["keyword_fraction_baseline"] = float(base)
print(json.dumps({k: out[k] for k in ("keyword_fraction_in_top5_cls_attention", "keyword_fraction_baseline", "cls_top_tokens_by_class")}, ensure_ascii=False)[:1500])
from src.transformer import unit_tests
out["unit_tests"] = unit_tests(); save_json(out, "part3_eval.json")
