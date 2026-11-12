"""Part 2 / Step 3: dataset preparation (500 sentences, rule-based POS + gazetteer BIO NER, stratified 70/15/15 split)."""
import sys, random
sys.path.insert(0, __file__.rsplit("/src/", 1)[0])
from src.common import *
from src import lexicons as L

TOPICS = {"پاکستان": ("Pakistan", 200), "کھیل": ("Sports", 150), "فن فنکار": ("Entertainment", 150)}


def build():
    rng = random.Random(SEED)
    meta = load_meta(); lines = open(P("cleaned.txt"), encoding="utf8").read().splitlines()
    pool = {k: [] for k in TOPICS}
    for m in meta:
        if m["category"] in TOPICS:
            for ln in range(m["cleaned_line_start"], m["cleaned_line_end"] + 1):
                toks = lines[ln - 1].split()
                if 6 <= len(toks) <= 35: pool[m["category"]].append((ln, m["doc_id"], toks))
    data = []
    for cat, (en, n) in TOPICS.items():
        for ln, doc, toks in rng.sample(pool[cat], n):
            data.append(dict(line=ln, doc=doc, topic=en, tokens=toks, pos=L.pos_tag(toks), ner=L.ner_tag(toks)))
    # stratified 70/15/15 split by topic
    split = {"train": [], "val": [], "test": []}
    for en in [v[0] for v in TOPICS.values()]:
        items = [d for d in data if d["topic"] == en]; rng.shuffle(items)
        a, b = int(.7 * len(items)), int(.85 * len(items))
        split["train"] += items[:a]; split["val"] += items[a:b]; split["test"] += items[b:]
    for k in split: rng.shuffle(split[k])
    return split


def write_conll(items, key, path):
    with open(path, "w", encoding="utf8") as f:
        for d in items:
            for t, g in zip(d["tokens"], d[key]): f.write(f"{t}\t{g}\n")
            f.write("\n")


def run():
    split = build()
    for key in ("pos", "ner"):
        for sp, items in split.items(): write_conll(items, key, P("data", f"{key}_{sp}.conll"))
    out = {"sizes": {k: len(v) for k, v in split.items()}, "topic_dist": {}, "pos_dist": {}, "ner_dist": {}}
    for sp, items in split.items():
        out["topic_dist"][sp] = dict(Counter(d["topic"] for d in items))
        out["pos_dist"][sp] = dict(Counter(t for d in items for t in d["pos"]))
        out["ner_dist"][sp] = dict(Counter(t for d in items for t in d["ner"]))
    allit = sum(split.values(), [])
    ents = Counter(); 
    for d in allit:
        for t in d["ner"]:
            if t.startswith("B-"): ents[t[2:]] += 1
    out["entity_counts"] = dict(ents); out["n_tokens"] = sum(len(d["tokens"]) for d in allit)
    out["lexicon_sizes"] = L.LEXICON_SIZES; out["gazetteer_sizes"] = L.GAZ_COUNTS
    out["topic_min"] = min(Counter(d["topic"] for d in allit).values())
    save_json(out, "part2_data.json")
    json.dump({k: v for k, v in split.items()}, open(P("data", "split_annotated.json"), "w", encoding="utf8"), ensure_ascii=False)
    return out


if __name__ == "__main__":
    o = run()
    print(json.dumps(o, ensure_ascii=False, indent=1)[:3500])
