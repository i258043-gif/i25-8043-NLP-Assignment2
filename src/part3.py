"""Part 3: 5-class topic classification with a from-scratch Transformer encoder vs a BiLSTM."""
import sys, time, math, copy
sys.path.insert(0, __file__.rsplit("/src/", 1)[0])
from src.common import *
from src.transformer import TransformerClassifier
import torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

CATS = ["Politics", "Sports", "Economy", "International", "Health & Society"]
SEQ = 256
KEYWORDS = {   # Urdu keyword lists guided by the assignment's indicative keywords (tokens are matched exactly on cleaned.txt tokens)
    "Politics": "انتخابات الیکشن حکومت وزیر وزیراعظم پارلیمنٹ اسمبلی سینیٹ پارٹی سیاسی سیاست صدر ووٹ اپوزیشن امیدوار کابینہ آئین جماعت وزرا گورنر",
    "Sports": "کرکٹ میچ ٹیم کھلاڑی کھلاڑیوں سکور رنز وکٹ اننگز ٹیسٹ کپتان کوچ ٹورنامنٹ سیریز فائنل گیند بیٹنگ بولنگ ٹرافی کپ مقابلے",
    "Economy": "مہنگائی افراط بینک ڈالر ٹیکس بجٹ معیشت معاشی تجارت قیمت قیمتوں روپے قرض سرمایہ کاری برآمدات درآمدات مارکیٹ کمپنی تیل فیصد",
    "International": "اقوام متحدہ معاہدہ خارجہ دوطرفہ بین الاقوامی تنازع جنگ امریکہ روس چین یوکرین اسرائیل غزہ فلسطین نیٹو سفارتی سفیر مذاکرات پابندیاں",
    "Health & Society": "ہسپتال بیماری مریض ڈاکٹر صحت ویکسین سیلاب تعلیم اسکول طلبہ خواتین بچوں علاج وائرس کورونا سماجی ماحولیاتی آلودگی خاندان زندگی",
}
SECTION_DEFAULT = {"پاکستان": "Politics", "کھیل": "Sports", "آس پاس": "International", "ورلڈ": "International",
                   "سائنس": "Health & Society", "فن فنکار": "Health & Society"}
MIN_EVIDENCE = 3


def build_dataset():
    meta = load_meta(); lines = open(P("cleaned.txt"), encoding="utf8").read().splitlines()
    stoi = json.load(open(P("embeddings", "word2idx.json"), encoding="utf8")); V = len(stoi); PAD_ID = V
    kw = {c: set(w.split()) for c, w in KEYWORDS.items()}
    ids, labels, how, toks_all = [], [], Counter(), []
    for m in meta:
        toks = [t for ln in range(m["cleaned_line_start"], m["cleaned_line_end"] + 1) for t in lines[ln - 1].split()]
        score = {c: sum(1 for t in toks if t in kw[c]) for c in CATS}
        best = max(CATS, key=lambda c: score[c])
        if score[best] >= MIN_EVIDENCE: lab = best; how["keyword evidence"] += 1
        else: lab = SECTION_DEFAULT[m["category"]]; how["BBC-section fallback (weak keyword evidence)"] += 1
        t256 = toks[:SEQ]; ids.append([stoi.get(t, 0) for t in t256] + [PAD_ID] * (SEQ - len(t256)))
        labels.append(CATS.index(lab)); toks_all.append(t256)
    ids = np.array(ids); labels = np.array(labels)
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(labels))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=labels[rest], random_state=SEED)
    sec = Counter((m["category_en"], CATS[l]) for m, l in zip(meta, labels))
    info = dict(n=len(labels), how=dict(how), dist={s: {CATS[c]: int((labels[i] == c).sum()) for c in range(5)} for s, i in (("all", idx), ("train", tr), ("val", va), ("test", te))},
                sizes=dict(train=len(tr), val=len(va), test=len(te)), crosstab={f"{a} -> {b}": n for (a, b), n in sorted(sec.items())},
                seq_len=SEQ, vocab=V, pad_id=PAD_ID, tokens_truncated=int(sum(1 for m in meta if m["n_tokens"] > SEQ)))
    np.savez(P("data", "cls_dataset.npz"), ids=ids, labels=labels, train=tr, val=va, test=te)
    json.dump(toks_all, open(P("data", "cls_tokens.json"), "w", encoding="utf8"), ensure_ascii=False)
    save_json(info, "part3_data.json"); return info


def load_ds():
    d = np.load(P("data", "cls_dataset.npz")); return d["ids"], d["labels"], d["train"], d["val"], d["test"]


# ----------------------------------------------------------------------------- BiLSTM classifier (for the comparison)
class BiLSTMClassifier(nn.Module):
    def __init__(self, emb, pad_id, n_classes=5, hidden=128):
        super().__init__()
        E = np.vstack([emb, np.zeros((1, emb.shape[1]), np.float32)])            # extra PAD row
        self.emb = nn.Embedding.from_pretrained(torch.tensor(E), freeze=False, padding_idx=pad_id); self.pad_id = pad_id
        self.lstm = nn.LSTM(E.shape[1], hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)
        self.drop = nn.Dropout(0.5); self.fc = nn.Linear(2 * hidden, n_classes)

    def forward(self, ids):
        lens = (ids != self.pad_id).sum(1).clamp(min=1)
        pk = pack_padded_sequence(self.emb(ids), lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(pk)                                                 # h: [layers*2, B, H]
        return self.fc(self.drop(torch.cat([h[-2], h[-1]], dim=1)))


def cosine_warmup(opt, warm, total):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


def run_epochs(kind, epochs=20, bs=16, budget=(int(os.environ["BUDGET"]) if "BUDGET" in os.environ else None)):
    set_seed(); ids, y, tr, va, te = load_ds(); info = load_json("part3_data.json"); pad = info["pad_id"]; V = info["vocab"]
    X = torch.tensor(ids); Y = torch.tensor(y)
    if kind == "transformer":
        model = TransformerClassifier(V + 1, pad, 5)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        sched = cosine_warmup(opt, 50, epochs * math.ceil(len(tr) / bs))
    else:
        model = BiLSTMClassifier(np.load(P("embeddings", "embeddings_w2v.npy")), pad, 5)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); sched = None
    ck = P("results", f"ck_p3_{kind}.pt"); st = dict(ep=0, hist=dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[], epoch_time=[]), best=None, best_val=-1, best_ep=0)
    if os.path.exists(ck):
        c = torch.load(ck, weights_only=False); model.load_state_dict(c["m"]); opt.load_state_dict(c["o"]); st = c["st"]
        if sched: sched.load_state_dict(c["sch"])
    t0 = time.time()
    def evaluate(idx):
        model.eval(); L, C = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                b = idx[i:i + 64]; o = model(X[b]); L += F.cross_entropy(o, Y[b], reduction="sum").item(); C += int((o.argmax(1) == Y[b]).sum())
        return L / len(idx), C / len(idx)
    while st["ep"] < epochs:
        if budget and time.time() - t0 > budget: print(f"UNFINISHED {kind} at epoch {st['ep']}"); return False
        rng = np.random.default_rng(SEED + st["ep"]); perm = rng.permutation(tr); model.train(); te0 = time.time(); L = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; loss = F.cross_entropy(model(X[b]), Y[b]); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if sched: sched.step()
            L += loss.item() * len(b)
        dt = time.time() - te0; vl, vacc = evaluate(va); _, tracc = evaluate(tr[:]) if False else (0, 0)
        trl, tra = evaluate(tr)                                                   # training loss/accuracy in eval mode
        h = st["hist"]; h["train_loss"].append(trl); h["train_acc"].append(tra); h["val_loss"].append(vl); h["val_acc"].append(vacc); h["epoch_time"].append(dt)
        st["ep"] += 1
        if vacc > st["best_val"]: st["best_val"], st["best_ep"], st["best"] = vacc, st["ep"], copy.deepcopy(model.state_dict())
        print(f"[{kind}] ep{st['ep']} train {trl:.3f}/{tra:.3f} val {vl:.3f}/{vacc:.3f} {dt:.0f}s", flush=True)
        torch.save(dict(m=model.state_dict(), o=opt.state_dict(), st=st, sch=sched.state_dict() if sched else None), ck)
    save_json(dict(hist=st["hist"], best_ep=st["best_ep"], best_val=st["best_val"]), f"p3_{kind}.json")
    torch.save(dict(state=st["best"], best_epoch=st["best_ep"], config=dict(d_model=128, heads=4, d_k=32, d_ff=512, layers=4, vocab=V + 1, pad_id=pad) if kind == "transformer" else {}),
               P("models", "transformer_cls.pt" if kind == "transformer" else "bilstm_cls.pt"))
    return True


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "data": print(json.dumps(build_dataset(), ensure_ascii=False, indent=1))
    else: run_epochs(a)
