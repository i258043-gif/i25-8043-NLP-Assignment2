"""Part 2 experiments: frozen vs fine-tuned, ablations A1-A4, evaluation, error analysis."""
import sys, time
sys.path.insert(0, __file__.rsplit("/src/", 1)[0])
from src.seqlabel import *

SEEDS = [42, 43]
CFGS = {   # name -> (task, kwargs)
    "pos_ft": ("pos", {}), "pos_frozen": ("pos", dict(freeze=True)),
    "pos_A1_unidir": ("pos", dict(bidirectional=False)), "pos_A2_nodrop": ("pos", dict(dropout=0.0)), "pos_A3_randinit": ("pos", dict(rand=True)),
    "ner_ft": ("ner", {}), "ner_frozen": ("ner", dict(freeze=True)),
    "ner_A1_unidir": ("ner", dict(bidirectional=False)), "ner_A2_nodrop": ("ner", dict(dropout=0.0)), "ner_A3_randinit": ("ner", dict(rand=True)),
    "ner_A4_softmax": ("ner", dict(crf=False)),
}


def _data():
    stoi = json.load(open(P("embeddings", "word2idx.json"), encoding="utf8")); E = np.load(P("embeddings", "embeddings_w2v.npy"))
    return stoi, E, load_split()


def run_all(budget=None):
    t0 = time.time(); stoi, E, split = _data()
    for seed in SEEDS:
        for name, (task, kw) in CFGS.items():
            f = P("results", "p2_runs", f"{name}_{seed}.json")
            if os.path.exists(f): continue
            if budget and time.time() - t0 > budget: print("UNFINISHED - run again"); return False
            kw = dict(kw); rand = kw.pop("rand", False)
            emb = E
            if rand:
                g = np.random.default_rng(seed); emb = g.normal(0, E.std(), E.shape).astype(np.float32)   # random initialisation (A3)
            r = run_training(task, emb, stoi, split, seed=seed, **kw)
            rec = dict(val=r["val"], test=r["test"], hist=r["hist"], best_epoch=r["best_epoch"], epochs_run=r["epochs_run"],
                       gold=r["gold"], pred=r["pred"], vgold=r["vgold"], vpred=r["vpred"])
            json.dump(rec, open(f, "w", encoding="utf8"), ensure_ascii=False)
            if seed == SEEDS[0] and name in ("pos_ft", "ner_ft"):
                torch.save(dict(state=r["model"].state_dict(), tags=r["tags"], config=dict(name=name, hidden=128, layers=2, dropout=0.5, bidirectional=True,
                                                                                         crf=name.startswith("ner"), embedding="w2v C3 (fine-tuned)")),
                           P("models", "bilstm_pos.pt" if name == "pos_ft" else "bilstm_ner.pt"))
            print(name, seed, r["val"], r["test"], r["best_epoch"], f"{time.time()-t0:.0f}s", flush=True)
    return True


if __name__ == "__main__":
    if sys.argv[1] == "run": run_all(float(sys.argv[2]) if len(sys.argv) > 2 else None)
