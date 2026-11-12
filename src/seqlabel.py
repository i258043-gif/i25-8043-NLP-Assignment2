"""BiLSTM sequence labeller (POS: softmax / NER: CRF or softmax), CRF with Viterbi, conlleval-style scoring."""
import sys, time, copy
sys.path.insert(0, __file__.rsplit("/src/", 1)[0])
from src.common import *
import torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.lexicons import TAGS, NER_TAGS


# --------------------------------------------------------------------------- CRF
class CRF(nn.Module):
    """Linear-chain CRF with a learnable tag-transition matrix (plus start/end scores); Viterbi decoding."""
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.trans = nn.Parameter(torch.randn(n, n) * 0.01)       # trans[i, j] = score of moving from tag i to tag j
        self.start = nn.Parameter(torch.zeros(n)); self.end = nn.Parameter(torch.zeros(n))

    def gold_score(self, em, tags, mask):
        B, T, _ = em.shape; m = mask.float()
        sc = self.start[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, T):
            step = self.trans[tags[:, t - 1], tags[:, t]] + em[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            sc = sc + step * m[:, t]
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        return sc + self.end[last]

    def log_partition(self, em, mask):
        B, T, n = em.shape
        alpha = self.start + em[:, 0]
        for t in range(1, T):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans.unsqueeze(0) + em[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)          # padding positions leave alpha unchanged
        return torch.logsumexp(alpha + self.end, dim=1)

    def nll(self, em, tags, mask):
        return (self.log_partition(em, mask) - self.gold_score(em, tags, mask)).mean()

    @torch.no_grad()
    def viterbi(self, em, mask):
        B, T, n = em.shape; lens = mask.sum(1).tolist()
        score = self.start + em[:, 0]; bps = []
        for t in range(1, T):
            cand = score.unsqueeze(2) + self.trans.unsqueeze(0)               # [B, from, to]
            best, bp = cand.max(1)
            new = best + em[:, t]
            keep = mask[:, t].unsqueeze(1)
            score = torch.where(keep, new, score); bps.append(torch.where(keep, bp, torch.arange(n).unsqueeze(0).expand(B, n)))
        score = score + self.end; out = []
        for b in range(B):
            tag = int(score[b].argmax()); path = [tag]
            for t in range(lens[b] - 1, 0, -1):
                tag = int(bps[t - 1][b, tag]); path.append(tag)
            out.append(path[::-1])
        return out


# --------------------------------------------------------------------------- model
class BiLSTMTagger(nn.Module):
    def __init__(self, emb, n_tags, hidden=128, layers=2, dropout=0.5, bidirectional=True, freeze=False, crf=False):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb, dtype=torch.float32), freeze=freeze, padding_idx=None)
        self.drop_in = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb.shape[1], hidden, num_layers=layers, bidirectional=bidirectional, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)              # dropout between the two LSTM layers
        self.drop_out = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), n_tags)       # h = [h_fwd || h_bwd] -> tag scores
        self.crf = CRF(n_tags) if crf else None

    def emissions(self, x, lens):
        e = self.drop_in(self.emb(x))
        pk = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(pk)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        return self.fc(self.drop_out(h))

    def loss(self, x, y, lens):
        em = self.emissions(x, lens); mask = torch.arange(x.size(1))[None, :] < lens[:, None]
        if self.crf is not None: return self.crf.nll(em, y.clamp(min=0), mask)
        return F.cross_entropy(em.reshape(-1, em.size(-1)), y.reshape(-1), ignore_index=-100)   # padding (-100) is ignored

    @torch.no_grad()
    def predict(self, x, lens):
        em = self.emissions(x, lens); mask = torch.arange(x.size(1))[None, :] < lens[:, None]
        if self.crf is not None: return self.crf.viterbi(em, mask)
        pr = em.argmax(-1); return [pr[b, :lens[b]].tolist() for b in range(x.size(0))]


# --------------------------------------------------------------------------- data
def load_split():
    return json.load(open(P("data", "split_annotated.json"), encoding="utf8"))


def encode_items(items, key, stoi, tag2i):
    return [([stoi.get(t, 0) for t in d["tokens"]], [tag2i[g] for g in d[key]]) for d in items]


def batches(data, bs, shuffle, rng=None):
    idx = list(range(len(data)))
    if shuffle: rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [data[j] for j in idx[i:i + bs]]; lens = torch.tensor([len(c[0]) for c in chunk]); T = int(lens.max())
        x = torch.zeros(len(chunk), T, dtype=torch.long); y = torch.full((len(chunk), T), -100, dtype=torch.long)
        for k, (w, t) in enumerate(chunk): x[k, :len(w)] = torch.tensor(w); y[k, :len(t)] = torch.tensor(t)
        yield x, y, lens


# --------------------------------------------------------------------------- metrics
def get_chunks(seq):
    """conlleval chunk extraction over BIO tag strings -> set of (type, start, end)."""
    chunks, start, ctype, prev = [], None, None, "O"
    for i, tag in enumerate(list(seq) + ["O"]):
        pre, typ = (tag.split("-", 1) + [""])[:2] if tag != "O" else ("O", "")
        pprev, ptyp = (prev.split("-", 1) + [""])[:2] if prev != "O" else ("O", "")
        end = pprev != "O" and (pre in ("B", "O") or typ != ptyp)
        begin = pre != "O" and (pre == "B" or pprev == "O" or typ != ptyp)
        if end: chunks.append((ptyp, start, i - 1))
        if begin: start = i
        prev = tag
    return set(chunks)


def conlleval(gold_seqs, pred_seqs):
    """Entity-level precision/recall/F1 per type and overall (exact-match chunks), like the CoNLL conlleval script."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold_seqs, pred_seqs):
        G, Pp = get_chunks(g), get_chunks(p)
        for c in G & Pp: tp[c[0]] += 1
        for c in Pp - G: fp[c[0]] += 1
        for c in G - Pp: fn[c[0]] += 1
    out = {}
    for t in sorted(set(tp) | set(fp) | set(fn)) + ["OVERALL"]:
        a, b, c = (sum(tp.values()), sum(fp.values()), sum(fn.values())) if t == "OVERALL" else (tp[t], fp[t], fn[t])
        pr = a / (a + b) if a + b else 0.0; rc = a / (a + c) if a + c else 0.0
        out[t] = dict(precision=pr, recall=rc, f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0, support=a + c)
    return out


def pos_metrics(gold, pred, labels):
    from sklearn.metrics import f1_score
    g = [x for s in gold for x in s]; p = [x for s in pred for x in s]
    present = sorted(set(g))
    return dict(acc=float(np.mean([a == b for a, b in zip(g, p)])), macro_f1=float(f1_score(g, p, labels=present, average="macro", zero_division=0)))


# --------------------------------------------------------------------------- training
def run_training(task, emb, stoi, split, seed=SEED, freeze=False, bidirectional=True, dropout=0.5, crf=None, max_epochs=80,
                 patience=5, min_epochs=20, hidden=128, bs=16, verbose=False):
    """task: 'pos' or 'ner'. Returns dict with best model state, curves, predictions."""
    set_seed(seed); rng = random.Random(seed)
    tags = TAGS if task == "pos" else NER_TAGS; t2i = {t: i for i, t in enumerate(tags)}
    D = {k: encode_items(split[k], task, stoi, t2i) for k in ("train", "val", "test")}
    use_crf = (task == "ner") if crf is None else crf
    model = BiLSTMTagger(emb, len(tags), hidden=hidden, bidirectional=bidirectional, dropout=dropout, freeze=freeze, crf=use_crf)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)

    def evaluate(data, split_items):
        model.eval(); preds = []
        for x, y, lens in batches(data, 32, False): preds += model.predict(x, lens)
        gold = [[tags[i] for i in d[1]] for d in data]; pred = [[tags[i] for i in p] for p in preds]
        m = pos_metrics(gold, pred, tags) if task == "pos" else {"f1": conlleval(gold, pred)["OVERALL"]["f1"]}
        return m, gold, pred

    def val_loss():
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for x, y, lens in batches(D["val"], 32, False): tot += model.loss(x, y, lens).item() * len(lens); n += len(lens)
        return tot / n

    hist = dict(train_loss=[], val_loss=[], val_f1=[]); best, best_f1, bad, best_ep = None, -1, 0, 0
    for ep in range(max_epochs):
        model.train(); tot, n = 0.0, 0
        for x, y, lens in batches(D["train"], bs, True, rng):
            loss = model.loss(x, y, lens); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); tot += loss.item() * len(lens); n += len(lens)
        vm, _, _ = evaluate(D["val"], split["val"]); vf = vm["macro_f1"] if task == "pos" else vm["f1"]
        hist["train_loss"].append(tot / n); hist["val_loss"].append(val_loss()); hist["val_f1"].append(vf)
        if verbose: print(f"ep{ep+1} tr={tot/n:.3f} va={hist['val_loss'][-1]:.3f} f1={vf:.3f}")
        if vf > best_f1 + 1e-9: best_f1, bad, best, best_ep = vf, 0, copy.deepcopy(model.state_dict()), ep + 1
        else:
            bad += 1
            if bad >= patience and ep + 1 >= min_epochs: break                                   # early stopping on validation F1
    model.load_state_dict(best)
    vm, vg, vp = evaluate(D["val"], split["val"]); tm, tg, tp = evaluate(D["test"], split["test"])
    return dict(model=model, val=vm, test=tm, gold=tg, pred=tp, vgold=vg, vpred=vp, hist=hist, best_epoch=best_ep, epochs_run=len(hist["train_loss"]), tags=tags)


# --------------------------------------------------------------------------- unit checks
def crf_selfcheck():
    """Brute-force verification of the CRF partition function and Viterbi decoder on a tiny random problem."""
    import itertools
    torch.manual_seed(0); n, T = 3, 4; crf = CRF(n); crf.trans.data = torch.randn(n, n); crf.start.data = torch.randn(n); crf.end.data = torch.randn(n)
    em = torch.randn(2, T, n); mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    res = {}
    for b in range(2):
        L = int(mask[b].sum()); best, allsc = (-1e9, None), []
        for path in itertools.product(range(n), repeat=L):
            s = crf.start[path[0]] + em[b, 0, path[0]] + sum(crf.trans[path[t - 1], path[t]] + em[b, t, path[t]] for t in range(1, L)) + crf.end[path[-1]]
            allsc.append(float(s)); best = max(best, (float(s), path))
        res[b] = dict(logZ_brute=float(np.log(np.sum(np.exp(allsc)))), best_brute=best[1])
    lz = crf.log_partition(em, mask); vit = crf.viterbi(em, mask)
    return {b: dict(logZ_crf=float(lz[b]), logZ_brute=res[b]["logZ_brute"], viterbi=tuple(vit[b]), brute=res[b]["best_brute"]) for b in range(2)}
