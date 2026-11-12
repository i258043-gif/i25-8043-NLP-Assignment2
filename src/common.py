"""Shared utilities: paths, seeds, corpus loading, vocabulary, Urdu plotting helpers."""
import os, json, random, time, re
from collections import Counter
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = lambda *a: os.path.join(ROOT, *a)
for d in ("embeddings", "models", "data", "results", "figures"):
    os.makedirs(P(d), exist_ok=True)

SEED = 42
UNK, PAD = "<UNK>", "<PAD>"
VOCAB_SIZE = 10000          # most frequent tokens kept (rest -> <UNK>)


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


set_seed()
torch.set_num_threads(max(1, os.cpu_count() or 1))

# ---------- Urdu-capable plotting ----------
_FONT = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"
if os.path.exists(_FONT):
    font_manager.fontManager.addfont(_FONT)
    plt.rcParams["font.family"] = ["FreeSerif", "DejaVu Sans"]
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titlesize": 12,
                     "axes.labelsize": 10, "figure.autolayout": True})
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _rs = arabic_reshaper.ArabicReshaper(configuration={"delete_harakat": True})

    def ur(s):
        """Shape + reorder an Urdu string so matplotlib draws it correctly."""
        return get_display(_rs.reshape(s))
except Exception:                                     # pragma: no cover
    def ur(s):
        return s


def log(msg, fh=None):
    print(msg, flush=True)
    if fh: fh.write(msg + "\n"); fh.flush()


# ---------- corpus ----------
def load_clean_sentences():
    with open(P("cleaned.txt"), encoding="utf8") as f:
        return [l.split() for l in f.read().splitlines() if l.strip()]


def load_raw_docs():
    with open(P("raw.txt"), encoding="utf8") as f:
        return [l.split() for l in f.read().splitlines() if l.strip()]


def load_meta():
    return json.load(open(P("Metadata.json"), encoding="utf8"))["documents"]


def build_vocab(seqs, size=VOCAB_SIZE):
    """seqs: iterable of token lists. Index 0 = <UNK>; then `size` most frequent tokens."""
    cnt = Counter(t for s in seqs for t in s)
    words = [w for w, _ in cnt.most_common(size)]
    itos = [UNK] + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos, cnt


def encode(seq, stoi):
    return [stoi.get(t, 0) for t in seq]


def cosine_topk(M, qvec, k=10, exclude=(), normed=False):
    """Top-k cosine neighbours of qvec among rows of M (dense np or torch)."""
    if not normed:
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    else:
        Mn = M
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    sims = Mn @ q
    for e in exclude:
        sims[e] = -9
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def save_json(obj, name):
    json.dump(obj, open(P("results", name), "w", encoding="utf8"), ensure_ascii=False, indent=1)


def load_json(name):
    return json.load(open(P("results", name), encoding="utf8"))
