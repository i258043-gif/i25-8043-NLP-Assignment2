"""Part 1 - Word embeddings: TF-IDF, PPMI, Skip-gram (negative sampling), evaluation."""
import sys, time, math
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, __file__.rsplit("/src/", 1)[0])
from src.common import *

WINDOW, NEG, EPOCHS, BATCH, LR, DIM = 5, 10, 25, 1024, 1e-3, 100
MINC = 20      # evaluation only: candidate neighbours must occur >= MINC times (standard min_count filter)

QUERY_WORDS = {"Pakistan": "پاکستان", "Hukumat": "حکومت", "Adalat": "عدالت", "Maeeshat": "معیشت",
               "Fauj": "فوج", "Sehat": "صحت", "Taleem": "تعلیم", "Aabadi": "آبادی"}
PPMI_QUERIES = ["پاکستان", "انڈیا", "حکومت", "کرکٹ", "فلم", "پولیس", "ٹیم", "عدالت", "فوج", "صحت", "چین", "ڈاکٹر"]
FOUR_COND_QUERIES = ["پاکستان", "حکومت", "عدالت", "فوج", "صحت"]

# 10 analogies a:b::c:?  (fixed a priori from vocabulary frequency, before any evaluation)
ANALOGIES = [("پاکستان", "پاکستانی", "انڈیا", "انڈین"), ("امریکہ", "امریکی", "آسٹریلیا", "آسٹریلوی"),
             ("چین", "چینی", "روس", "روسی"), ("بیٹا", "بیٹی", "بھائی", "بہن"),
             ("پہلی", "پہلا", "دوسری", "دوسرا"), ("بڑی", "بڑا", "نئی", "نیا"),
             ("مرد", "عورت", "شوہر", "بیوی"), ("لاہور", "پنجاب", "کراچی", "سندھ"),
             ("جاتا", "جاتی", "کرتا", "کرتی"), ("مردوں", "مرد", "بچوں", "بچہ")]

# 20 manually labelled related word pairs (query -> expected related word) for MRR
LABELLED_PAIRS = [("کرکٹ", "ٹیم"), ("میچ", "رنز"), ("وکٹ", "اننگز"), ("کپتان", "کھلاڑی"), ("فلم", "اداکار"),
                  ("فلم", "گانے"), ("حکومت", "وزیر"), ("عدالت", "فیصلہ"), ("فوج", "جنگ"), ("صحت", "ہسپتال"),
                  ("ڈاکٹر", "ہسپتال"), ("علاج", "بیماری"), ("تعلیم", "یونیورسٹی"), ("ڈالر", "روپے"),
                  ("معیشت", "مہنگائی"), ("انڈیا", "پاکستان"), ("چین", "روس"), ("پاکستان", "پاکستانی"),
                  ("بیٹی", "بیٹا"), ("شوہر", "بیوی")]

# semantic categories for the t-SNE colouring (hand-built lexicon)
TSNE_CATS = {
    "Politics / state": "حکومت وزیر پولیس سیاسی پارٹی صدر قانون فوج خلاف الزام فیصلہ جنگ عالمی سابق بیان اعلان عدالت خان شاہ علی احمد محمد",
    "Sports": "ٹیم کرکٹ میچ رنز ٹیسٹ اننگز کھلاڑی سیریز گیند وکٹ بیٹنگ جیت ڈے کپ ون کھیل مقابلے",
    "Geography / places": "پاکستان انڈیا انڈین پاکستانی چین روس عرب امریکہ امریکی آسٹریلیا آسٹریلوی سعودی یوکرین پنجاب کراچی لاہور شہر علاقے ملک ممالک دنیا یہاں وہاں ورلڈ اسلام آباد",
    "Entertainment / media": "فلم فلموں گانے ویڈیو کہانی کردار شادی میڈیا سوشل",
    "Society / economy": "ڈالر ٹیکس فیصد تیل کمپنی ڈاکٹر خواتین بچے لوگ افراد خاندان زندگی تعلیم صحت ہزار لاکھ",
    "Function words": "کے میں کی ہے سے اور کہ کا نے کو اس ہیں پر بھی ایک ان یہ کر تھا وہ نہیں کیا لیے ہو تو تھے گیا کرنے تھی ہی بعد جو لیکن اپنے جس ساتھ جب رہے ہوئے تک بہت کرتے اپنی والے انھیں انھوں کوئی کچھ ہم آپ گئی ہونے رہا یا گئے گا اب نہ کسی پہلے رہی ہوا جا سب دی ہوں سی والی دو اگر کم پھر صرف جائے گی ہوئی ہوتا مجھے تھیں اسے جاتا آئی سکتا ایسے جن لے ایسا جہاں مگر جانے ہوتی کیونکہ کئی دونوں ہر پاس اسی تاہم کیے لیا خود طرف جیسے دے آ کریں والا جبکہ ایسی کیوں ہمیں بے بن میرے",
}


# ------------------------------------------------------------------ 1.1 TF-IDF
def tfidf_part(logf):
    meta = load_meta(); sents = load_clean_sentences()
    stoi, itos, cnt = build_vocab(sents)
    V = len(itos); N = len(meta)
    rows, cols, vals = [], [], []
    for d, m in enumerate(meta):
        c = Counter()
        for s in sents[m["cleaned_line_start"] - 1: m["cleaned_line_end"]]:
            c.update(stoi.get(t, 0) for t in s)
        for w, n in c.items():
            rows.append(w); cols.append(d); vals.append(n)
    TD = sp.csr_matrix((vals, (rows, cols)), shape=(V, N), dtype=np.float32)     # term-document counts
    df = np.asarray((TD > 0).sum(1)).ravel()
    idf = np.log(N / (1.0 + df)).astype(np.float32)
    TFIDF = (TD.multiply(idf[:, None])).toarray().astype(np.float32)
    np.save(P("embeddings", "tfidf_matrix.npy"), TFIDF)
    json.dump(stoi, open(P("embeddings", "word2idx.json"), "w", encoding="utf8"), ensure_ascii=False)
    log(f"term-document matrix {TD.shape}, nnz={TD.nnz}; TF-IDF matrix saved {TFIDF.shape}; vocab cap {VOCAB_SIZE} (+<UNK>)", logf)
    cats = sorted({m["category_en"] for m in meta})
    top = {}
    for c in cats:
        ids = [i for i, m in enumerate(meta) if m["category_en"] == c]
        score = TFIDF[:, ids].mean(1); score[0] = -9                       # ignore <UNK>
        top[c] = [itos[i] for i in np.argsort(-score)[:10]]
        log(f"  top-10 TF-IDF words [{c}] ({len(ids)} docs): " + " ".join(top[c]), logf)
    return stoi, itos, sents, dict(top=top, n_docs=N, vocab=V)


# ------------------------------------------------------------------ 1.2 PPMI
def cooccurrence(seqs_idx, V, k=WINDOW):
    r, c = [], []
    for s in seqs_idx:
        a = np.asarray(s, dtype=np.int64)
        for o in range(1, k + 1):
            if len(a) <= o: break
            x, y = a[:-o], a[o:]
            m = (x > 0) & (y > 0)                                           # skip <UNK>
            r.append(x[m]); c.append(y[m])
    r = np.concatenate(r); c = np.concatenate(c)
    C = sp.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(V, V)).tocsr()
    return (C + C.T).tocsr()                                               # symmetric window


def ppmi_from_counts(C):
    C = C.tocoo(); total = C.data.sum()
    rs = np.asarray(C.sum(1)).ravel(); cs = np.asarray(C.sum(0)).ravel()
    pmi = np.log2((C.data * total) / (rs[C.row] * cs[C.col] + 1e-12))
    pmi = np.maximum(pmi, 0).astype(np.float32)
    M = sp.coo_matrix((pmi, (C.row, C.col)), shape=C.shape).tocsr(); M.eliminate_zeros()
    return M


def sparse_neighbors(Mn, i, k=5, valid=None):
    sims = (Mn @ Mn[i].T).toarray().ravel()
    sims[0] = -9; sims[i] = -9
    if valid is not None: sims[~valid] = -9
    idx = np.argpartition(-sims, k)[:k]
    return idx[np.argsort(-sims[idx])], sims[idx[np.argsort(-sims[idx])]]


def l2norm_sparse(M):
    n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()) + 1e-9
    return sp.diags(1.0 / n) @ M


def ppmi_part(stoi, itos, sents, logf):
    V = len(itos)
    C = cooccurrence([encode(s, stoi) for s in sents], V)
    M = ppmi_from_counts(C)
    log(f"co-occurrence nnz={C.nnz}, PPMI nnz={M.nnz} ({100*M.nnz/V/V:.2f}% dense), window k={WINDOW} symmetric", logf)
    Mn = l2norm_sparse(M).tocsr()
    nn_res = {}
    for w in PPMI_QUERIES:
        idx, sims = sparse_neighbors(Mn, stoi[w], 5)
        nn_res[w] = [(itos[i], round(float(s), 3)) for i, s in zip(idx, sims)]
        log(f"  PPMI NN {w}: " + " ".join(f"{a}({b})" for a, b in nn_res[w]), logf)
    # t-SNE of the 200 most frequent tokens (PPMI row vectors, cosine metric)
    from sklearn.manifold import TSNE
    top = list(range(1, 201))
    X = Mn[top].toarray()
    Y = TSNE(n_components=2, perplexity=25, metric="cosine", init="random", random_state=SEED, max_iter=1500).fit_transform(X)
    cat_of = {}
    for cname, ws in TSNE_CATS.items():
        for w in ws.split(): cat_of.setdefault(w, cname)
    colors = {"Politics / state": "#d62728", "Sports": "#2ca02c", "Geography / places": "#1f77b4",
              "Entertainment / media": "#9467bd", "Society / economy": "#ff7f0e",
              "Function words": "#7f7f7f", "Other": "#bcbd22"}
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for cname, col in colors.items():
        ids = [j for j, i in enumerate(top) if cat_of.get(itos[i], "Other") == cname]
        ax.scatter(Y[ids, 0], Y[ids, 1], c=col, s=38, label=f"{cname} ({len(ids)})", alpha=.85, edgecolors="k", linewidths=.3)
        for j in ids: ax.annotate(ur(itos[top[j]]), Y[j], fontsize=8, xytext=(2, 2), textcoords="offset points")
    ax.set_title("t-SNE of the 200 most frequent tokens (PPMI vectors, cosine), coloured by semantic category")
    ax.set_xlabel("t-SNE dimension 1"); ax.set_ylabel("t-SNE dimension 2"); ax.legend(loc="best", fontsize=8)
    fig.savefig(P("figures", "tsne_ppmi_top200.png")); plt.close(fig)
    dense = M.toarray().astype(np.float32)
    np.save(P("embeddings", "ppmi_matrix.npy"), dense)
    log(f"PPMI matrix saved {dense.shape}; t-SNE figure saved", logf)
    return Mn, nn_res, {c: sum(1 for i in top if cat_of.get(itos[i], 'Other') == c) for c in list(colors)}


# ------------------------------------------------------------------ 2.1 Skip-gram negative sampling
class SkipGram(nn.Module):
    """Centre matrix V and context matrix U, both |V| x d."""
    def __init__(self, V, d):
        super().__init__()
        self.V = nn.Embedding(V, d); self.U = nn.Embedding(V, d)
        nn.init.uniform_(self.V.weight, -0.5 / d, 0.5 / d); nn.init.uniform_(self.U.weight, -0.5 / d, 0.5 / d)

    def forward(self, c, o, neg):                      # c:[B] o:[B] neg:[B,K]
        vc = self.V(c)                                 # [B,d]
        pos = F.logsigmoid((self.U(o) * vc).sum(-1))   # log sigma(u_o . v_c)
        ng = F.logsigmoid(-torch.bmm(self.U(neg), vc.unsqueeze(2)).squeeze(2)).sum(-1)   # sum_k log sigma(-u_w . v_c)
        return -(pos + ng).mean()

    def final(self):                                   # averaged embeddings 1/2 (V+U)
        return 0.5 * (self.V.weight + self.U.weight).detach().cpu().numpy()


def make_pairs(tok, sid, keep, rng, k=WINDOW):
    """tok/sid: flat arrays of token ids and sentence ids. Frequent-word subsampling then (centre, context) pairs
    inside a symmetric window of size k (never crossing a sentence/document boundary, never involving <UNK>)."""
    m = rng.random(len(tok)) < keep[tok]
    t, s = tok[m], sid[m]
    cs, os_ = [], []
    for o in range(1, k + 1):
        ok = (s[:-o] == s[o:]) & (t[:-o] > 0) & (t[o:] > 0)
        a, b = t[:-o][ok], t[o:][ok]
        cs += [a, b]; os_ += [b, a]
    return np.concatenate(cs), np.concatenate(os_)


def train_skipgram(seqs_idx, V, d=DIM, epochs=EPOCHS, batch=BATCH, lr=LR, tag="C3", logf=None, sub_t=1e-4, ckpt=None, budget=None):
    set_seed()
    tok = np.concatenate([np.asarray(s) for s in seqs_idx]).astype(np.int64)
    sid = np.concatenate([np.full(len(s), i) for i, s in enumerate(seqs_idx)]).astype(np.int64)
    freq = np.bincount(tok, minlength=V).astype(np.float64)
    f = freq / freq.sum()
    keep = np.minimum(1.0, np.sqrt(sub_t / np.maximum(f, 1e-12)) + sub_t / np.maximum(f, 1e-12))   # word2vec subsampling
    noise = freq ** 0.75; noise[0] = 0; noise /= noise.sum()                                       # P(w) ~ f(w)^(3/4)
    table = torch.from_numpy(np.random.choice(V, size=2_000_000, p=noise)).long()                  # unigram^(3/4) table
    model = SkipGram(V, d); opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, step, t0, ep0 = [], 0, time.time(), 0
    if ckpt and os.path.exists(ckpt):                      # resume from the last finished epoch
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); hist, step, ep0 = st["h"], st["s"], st["e"]
        log(f"[{tag}] resumed at epoch {ep0}", logf)
    for ep in range(ep0, epochs):
        if budget and time.time() - t0 > budget:
            return None, hist                               # out of wall-clock budget; call again to resume
        rng = np.random.default_rng(SEED + ep)
        c, o = make_pairs(tok, sid, keep, rng)
        perm = rng.permutation(len(c)); c = torch.from_numpy(c[perm]); o = torch.from_numpy(o[perm])
        run, nb = 0.0, 0
        for i in range(0, len(c) - batch + 1, batch):
            cb, ob = c[i:i + batch], o[i:i + batch]
            neg = table[torch.randint(0, len(table), (batch, NEG))]
            loss = model(cb, ob, neg)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item(); nb += 1; step += 1
            if step % 200 == 0: hist.append((step, run / nb))
        log(f"[{tag}] epoch {ep+1}/{epochs} pairs={len(c):,} loss={run/max(nb,1):.4f} ({time.time()-t0:.0f}s)", logf)
        if ckpt: torch.save(dict(m=model.state_dict(), o=opt.state_dict(), h=hist, s=step, e=ep + 1), ckpt)
    return model, hist


# ------------------------------------------------------------------ evaluation helpers
def normed(E): return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def dense_nn(En, stoi, itos, w, k=10, valid=None):
    i = stoi[w]; sims = En @ En[i]; sims[0] = -9; sims[i] = -9
    if valid is not None: sims[~valid] = -9
    idx = np.argsort(-sims)[:k]
    return [(itos[j], round(float(sims[j]), 3)) for j in idx]


def analogy_eval(En, stoi, itos, tests=ANALOGIES, k=3, valid=None):
    out = []
    for a, b, c, d in tests:
        q = En[stoi[b]] - En[stoi[a]] + En[stoi[c]]
        sims = En @ (q / (np.linalg.norm(q) + 1e-9)); sims[0] = -9
        if valid is not None: sims[~valid] = -9
        for w in (a, b, c): sims[stoi[w]] = -9
        idx = np.argsort(-sims)[:k]
        cands = [itos[j] for j in idx]
        out.append(dict(a=a, b=b, c=c, gold=d, top3=cands, top1_correct=cands[0] == d, top3_correct=d in cands))
    return out


def mrr_eval(rank_fn, stoi, pairs=LABELLED_PAIRS, valid=None):
    rr, missing = [], 0
    for a, b in pairs:
        if a not in stoi or b not in stoi: rr.append(0.0); missing += 1; continue
        sims = rank_fn(stoi[a]); sims[0] = -9; sims[stoi[a]] = -9
        if valid is not None: sims[~valid] = -9; sims[stoi[b]] = rank_fn(stoi[a])[stoi[b]]
        rank = 1 + int((sims > sims[stoi[b]]).sum())
        rr.append(1.0 / rank)
    return float(np.mean(rr)), missing, rr


# ------------------------------------------------------------------ main experiment (resumable stages)
def _prep():
    sents = load_clean_sentences(); stoi, itos, _ = build_vocab(sents)
    return sents, stoi, itos


def stage_A():
    logf = open(P("results", "part1_log.txt"), "a", encoding="utf8")
    stoi, itos, sents, tf = tfidf_part(logf)
    Mn, nn_res, tsne_counts = ppmi_part(stoi, itos, sents, logf)
    save_json(dict(tfidf=tf, ppmi_nn=nn_res, tsne_counts=tsne_counts), "p1_A.json")


def stage_train(tag):
    logf = open(P("results", "part1_log.txt"), "a", encoding="utf8")
    if tag == "C2":
        raw = load_raw_docs(); stoi_r, itos_r, _ = build_vocab(raw)
        json.dump(stoi_r, open(P("embeddings", "word2idx_raw.json"), "w", encoding="utf8"), ensure_ascii=False)
        m, h = train_skipgram([encode(s, stoi_r) for s in raw], len(itos_r), DIM, tag="C2", logf=logf, ckpt=P("results", "ck_C2.pt"), budget=(int(os.environ["BUDGET"]) if "BUDGET" in os.environ else None))
        if m is None: print("UNFINISHED - run again"); return
        np.save(P("embeddings", "embeddings_w2v_raw.npy"), m.final())
    else:
        sents, stoi, itos = _prep(); seqs = [encode(s, stoi) for s in sents]
        d = DIM if tag == "C3" else 200
        m, h = train_skipgram(seqs, len(itos), d, tag=tag, logf=logf, ckpt=P("results", f"ck_{tag}.pt"), budget=(int(os.environ["BUDGET"]) if "BUDGET" in os.environ else None))
        if m is None: print("UNFINISHED - run again"); return
        np.save(P("embeddings", "embeddings_w2v.npy" if tag == "C3" else "embeddings_w2v_d200.npy"), m.final())
        if tag == "C3": torch.save(m.state_dict(), P("models", "skipgram_c3.pt"))
    save_json(dict(hist=h), f"p1_{tag}.json")


def stage_eval():
    logf = open(P("results", "part1_eval_log.txt"), "w", encoding="utf8")
    sents, stoi, itos = _prep(); V = len(itos)
    res = load_json("p1_A.json")
    C = cooccurrence([encode(s, stoi) for s in sents], V); Mn = l2norm_sparse(ppmi_from_counts(C)).tocsr()
    E3, E2, E4 = (np.load(P("embeddings", f)) for f in ("embeddings_w2v.npy", "embeddings_w2v_raw.npy", "embeddings_w2v_d200.npy"))
    stoi_r = json.load(open(P("embeddings", "word2idx_raw.json"), encoding="utf8")); itos_r = [None] * len(stoi_r)
    for w, i in stoi_r.items(): itos_r[i] = w
    En3, En2, En4 = normed(E3), normed(E2), normed(E4)
    cnt = Counter(t for sn in sents for t in sn); cnt_r = Counter(t for d in load_raw_docs() for t in d)
    valid = np.array([i > 0 and cnt[w] >= MINC for i, w in enumerate(itos)])
    valid_r = np.array([i > 0 and cnt_r[w] >= MINC for i, w in enumerate(itos_r)])
    res["min_count_filter"] = dict(MINC=MINC, candidates_clean=int(valid.sum()), candidates_raw=int(valid_r.sum()))
    h3, h2, h4 = (load_json(f"p1_{t}.json")["hist"] for t in ("C3", "C2", "C4"))

    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(*zip(*h3), lw=1.4)
    ax.set_title("Skip-gram (C3, d=100) training loss"); ax.set_xlabel("Training step (batch of %d pairs)" % BATCH)
    ax.set_ylabel("Mean BCE loss (running epoch average)"); ax.grid(alpha=.3)
    fig.savefig(P("figures", "w2v_loss_curve.png")); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for h, lab in ((h3, "C3 cleaned d=100"), (h2, "C2 raw d=100"), (h4, "C4 cleaned d=200")): ax.plot(*zip(*h), label=lab, lw=1.3)
    ax.set_title("Skip-gram training loss for conditions C2, C3, C4"); ax.set_xlabel("Training step")
    ax.set_ylabel("Mean BCE loss"); ax.legend(); ax.grid(alpha=.3)
    fig.savefig(P("figures", "w2v_loss_all_conditions.png")); plt.close(fig)

    res["nn10"] = {k: dense_nn(En3, stoi, itos, w, 10, valid) for k, w in QUERY_WORDS.items()}
    for k, w in QUERY_WORDS.items():
        log(f"  top-10 NN {k} ({w}): " + " ".join(f"{a}({b})" for a, b in res["nn10"][k]), logf)
    res["analogies"] = analogy_eval(En3, stoi, itos, valid=valid)
    for t in res["analogies"]:
        log(f"  {t['a']} : {t['b']} :: {t['c']} : ?  gold={t['gold']} top3={t['top3']} top1={'OK' if t['top1_correct'] else '-'} top3={'OK' if t['top3_correct'] else '-'}", logf)
    log(f"  analogies correct: top-1 {sum(t['top1_correct'] for t in res['analogies'])}/10, in top-3 {sum(t['top3_correct'] for t in res['analogies'])}/10", logf)

    conds = {}
    mrr, miss, rr = mrr_eval(lambda i: (Mn @ Mn[i].T).toarray().ravel(), stoi, valid=valid)
    conds["C1 PPMI baseline"] = dict(mrr=mrr, missing=miss, rr=rr, dim="10,001 (sparse)", nn5={w: [itos[i] for i in sparse_neighbors(Mn, stoi[w], 5, valid)[0]] for w in FOUR_COND_QUERIES})
    mrr, miss, rr = mrr_eval(lambda i: En2 @ En2[i], stoi_r, valid=valid_r)
    conds["C2 Skip-gram raw (d=100)"] = dict(mrr=mrr, missing=miss, rr=rr, dim=100, nn5={w: [x for x, _ in dense_nn(En2, stoi_r, itos_r, w, 5, valid_r)] if w in stoi_r else [] for w in FOUR_COND_QUERIES},
                                           raw_vocab=len(itos_r), raw_vocab_with_punct=sum(1 for t in itos_r[1:] if re.search(r"[۔،’‘؟!:\"\(\)\-]", t)))
    mrr, miss, rr = mrr_eval(lambda i: En3 @ En3[i], stoi, valid=valid)
    conds["C3 Skip-gram cleaned (d=100)"] = dict(mrr=mrr, missing=miss, rr=rr, dim=100, nn5={w: [x for x, _ in dense_nn(En3, stoi, itos, w, 5, valid)] for w in FOUR_COND_QUERIES})
    mrr, miss, rr = mrr_eval(lambda i: En4 @ En4[i], stoi, valid=valid)
    conds["C4 Skip-gram cleaned (d=200)"] = dict(mrr=mrr, missing=miss, rr=rr, dim=200, nn5={w: [x for x, _ in dense_nn(En4, stoi, itos, w, 5, valid)] for w in FOUR_COND_QUERIES})
    res["conditions"] = conds
    for k, v in conds.items():
        log(f"  {k}: MRR={v['mrr']:.4f} (missing pairs={v['missing']})", logf)
        for w, nn5 in v["nn5"].items(): log(f"      {w}: {' '.join(nn5)}", logf)
    res["config"] = dict(window=WINDOW, K=NEG, epochs=EPOCHS, batch=BATCH, lr=LR, d=DIM, vocab=V, tokens=int(sum(len(s) for s in sents)), subsample_t=1e-4)
    res["loss_hist_c3"] = h3
    save_json(res, "part1.json")


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "A": stage_A()
    elif st == "eval": stage_eval()
    else: stage_train(st)
