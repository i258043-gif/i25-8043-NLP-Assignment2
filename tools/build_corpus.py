"""Build raw.txt, cleaned.txt and Metadata.json for the BBC Urdu corpus.
Sources (public GitHub scrapes of BBC Urdu, 6 sections, 100 stories each):
  Dev-Khuraim/WEB-Scraper  -> FetchedData.xlsx
  asb809/WEB-SCRAPER       -> BBC.xlsx
"""
import re, json, unicodedata, pandas as pd

CAT_EN = {"پاکستان":"Pakistan","آس پاس":"Region (Aas Paas)","ورلڈ":"World",
          "کھیل":"Sports","فن فنکار":"Entertainment","سائنس":"Science"}

a = pd.read_excel("WEB-Scraper/FetchedData.xlsx")
b = pd.read_excel("WEB-SCRAPER/BBC.xlsx").rename(columns={"Headline":"Title"})
a["source"], b["source"] = "Dev-Khuraim/WEB-Scraper", "asb809/WEB-SCRAPER"
df = pd.concat([a, b], ignore_index=True).dropna(subset=["Story","Title"])

def flat(s):                       # raw: only collapse line breaks / whitespace
    s = str(s).replace("\\n", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()

df["Title"] = df.Title.map(flat); df["Story"] = df.Story.map(flat)
df["key"] = df.Story.str.replace(r"\W+", "", regex=True).str[:300]
df = df.drop_duplicates("key").drop_duplicates("Title").reset_index(drop=True)
df = df[df.Story.str.split().str.len() >= 30].reset_index(drop=True)

MAP = {"ي":"ی","ى":"ی","ك":"ک","ہ":"ہ","ۀ":"ۂ"}
def norm_token_text(s):
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    out = []
    for ch in s:
        ch = MAP.get(ch, ch)
        cat = unicodedata.category(ch)
        if ch.isspace(): out.append(" ")
        elif cat == "Nd":           # Arabic-Indic / Persian digits -> ASCII
            out.append(str(unicodedata.digit(ch)))
        elif cat.startswith("L") and ("ARABIC" in unicodedata.name(ch, "")): out.append(ch)
        else: out.append(" ") if cat not in ("Mn","Cf") else None   # drop diacritics, ZWNJ/ZWJ
    return re.sub(r"\s+", " ", "".join(out)).strip()

SPLIT = re.compile(r"[۔؟!?\.\n]+")
def sentences(text):
    for part in SPLIT.split(text):
        t = norm_token_text(part)
        if len(t.split()) >= 3: yield t

raw_lines, clean_lines, meta = [], [], []
for i, r in df.iterrows():
    full = f"{r.Title}۔ {r.Story}"
    raw_lines.append(f"{r.Title} {r.Story}")
    sents = list(sentences(full))
    start = len(clean_lines) + 1
    clean_lines.extend(sents)
    meta.append({
        "doc_id": i, "title": r.Title, "category": r.Category,
        "category_en": CAT_EN[r.Category], "source": r.source,
        "raw_line": i + 1, "cleaned_line_start": start,
        "cleaned_line_end": len(clean_lines),
        "n_sentences": len(sents), "n_tokens": sum(len(s.split()) for s in sents)})

open("raw.txt","w",encoding="utf-8").write("\n".join(raw_lines)+"\n")
open("cleaned.txt","w",encoding="utf-8").write("\n".join(clean_lines)+"\n")
json.dump({"description":"BBC Urdu corpus. One entry per article; cleaned_line_start/end "
           "(1-indexed, inclusive) give the article's sentences in cleaned.txt (one sentence per line); "
           "raw_line is the article's line in raw.txt (one article per line).",
           "n_documents": len(meta), "documents": meta},
          open("Metadata.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)

from collections import Counter
toks = [t for l in clean_lines for t in l.split()]
print("docs",len(meta),"sentences",len(clean_lines),"tokens",len(toks),"vocab",len(set(toks)))
print(Counter(m["category_en"] for m in meta))
print("tokens in top10k:", sum(c for _,c in Counter(toks).most_common(10000))/len(toks))
