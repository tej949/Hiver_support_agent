"""Utilities for adapting a CSV export of Customer Support on Twitter.
This deliberately avoids assuming one exact Kaggle schema."""
import argparse, json, re
from pathlib import Path
import pandas as pd

def first_existing(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def prepare(csv_path, brand, out_path):
    df = pd.read_csv(csv_path, low_memory=False)
    text_col = first_existing(df, ["text", "tweet", "body"])
    author_col = first_existing(df, ["author_id", "user_id", "author"])
    if not text_col:
        raise ValueError("Could not find a text column. Expected one of: text, tweet, body")
    mask = df[text_col].fillna("").astype(str).str.contains(re.escape(brand), case=False, regex=True)
    sub = df.loc[mask].copy()
    # Keep a compact, inspectable candidate file. Full thread reconstruction depends on the exact export schema.
    rows = []
    for i, r in sub.iterrows():
        rows.append({"row_id": int(i), "text": str(r[text_col]), "author": str(r[author_col]) if author_col else ""})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} candidate rows to {out_path}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--brand", required=True)
    p.add_argument("--out", default="data/brand_candidates.jsonl")
    a=p.parse_args(); prepare(a.csv,a.brand,a.out)
if __name__ == "__main__": main()
