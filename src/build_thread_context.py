"""
Build thread-aware AmazonHelp conversations.

Uses the original TWCS raw file because the existing pair file does not retain
customer_tweet.in_response_to_tweet_id, which is required to reconstruct the
actual reply chain.

Output:
  data/processed/amazonhelp_thread_context.csv

Each row is one customer -> AmazonHelp reply pair, augmented with up to
MAX_PREVIOUS_TURNS actual previous turns from the same Twitter reply thread.
"""

from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw/twcs.csv")
PAIRS = Path("data/processed/amazonhelp_conversations.csv")
OUT = Path("data/processed/amazonhelp_thread_context.csv")

MAX_PREVIOUS_TURNS = 3


def norm_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def main():
    print("=" * 70)
    print("THREAD-AWARE CONTEXT RECONSTRUCTION")
    print("=" * 70)

    if not RAW.exists():
        raise FileNotFoundError(f"Missing raw dataset: {RAW}")
    if not PAIRS.exists():
        raise FileNotFoundError(f"Missing pair dataset: {PAIRS}")

    print(f"Loading raw TWCS: {RAW}")
    cols = [
        "tweet_id", "author_id", "inbound", "created_at", "text",
        "response_tweet_id", "in_response_to_tweet_id"
    ]
    raw = pd.read_csv(RAW, usecols=cols, dtype=str, low_memory=False)
    raw["tweet_id"] = raw["tweet_id"].map(norm_id)
    raw["in_response_to_tweet_id"] = raw["in_response_to_tweet_id"].map(norm_id)
    raw["author_id"] = raw["author_id"].map(norm_id)
    raw["inbound"] = raw["inbound"].fillna("")
    raw["text"] = raw["text"].fillna("")
    raw["created_at"] = raw["created_at"].fillna("")

    pairs = pd.read_csv(PAIRS, dtype=str, low_memory=False).fillna("")
    for c in pairs.columns:
        pairs[c] = pairs[c].map(lambda x: norm_id(x) if "id" in c else x)

    # Only AmazonHelp replies that are represented in the existing pair file.
    target_ids = set(pairs["brand_tweet_id"].map(norm_id))
    print(f"AmazonHelp reply pairs: {len(target_ids):,}")

    # Build a compact lookup for reply-chain traversal.
    # The raw dataset has ~2.8M rows; keeping only these fields is intentional.
    raw = raw.drop_duplicates("tweet_id", keep="first").set_index("tweet_id", drop=False)

    def row_for(tweet_id):
        if not tweet_id:
            return None
        try:
            r = raw.loc[tweet_id]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            return r
        except KeyError:
            return None

    # Existing pair rows are the authoritative current customer -> brand pairs.
    # Add the missing parent pointer for the customer tweet from raw data.
    pairs["customer_in_response_to"] = pairs["customer_tweet_id"].map(
        lambda x: (
            row_for(norm_id(x))["in_response_to_tweet_id"]
            if row_for(norm_id(x)) is not None else ""
        )
    )

    # Map brand reply id -> pair row index. This lets us recover the previous
    # customer turn when a current customer replied to a previous AmazonHelp tweet.
    pair_by_brand = {}
    for i, r in pairs.iterrows():
        bid = norm_id(r["brand_tweet_id"])
        if bid:
            pair_by_brand[bid] = i

    def get_previous_turns(customer_id):
        """
        Walk:
          current customer
          -> parent AmazonHelp tweet
          -> parent customer
          -> parent AmazonHelp tweet
          ...
        and return chronological previous customer/brand turns.
        """
        turns_rev = []
        seen = set()
        cur_customer = norm_id(customer_id)

        for _ in range(MAX_PREVIOUS_TURNS):
            crow = row_for(cur_customer)
            if crow is None:
                break

            parent_brand = norm_id(crow["in_response_to_tweet_id"])
            if not parent_brand or parent_brand in seen:
                break
            seen.add(parent_brand)

            # Parent AmazonHelp reply must exist in the pair set.
            if parent_brand not in pair_by_brand:
                break

            idx = pair_by_brand[parent_brand]
            prow = pairs.iloc[idx]

            turns_rev.append({
                "customer_text": clean_text(prow["customer_text"]),
                "brand_response": clean_text(prow["brand_response"]),
                "customer_tweet_id": norm_id(prow["customer_tweet_id"]),
                "brand_tweet_id": parent_brand,
                "customer_created_at": clean_text(prow["customer_created_at"]),
                "brand_created_at": clean_text(prow["brand_created_at"]),
            })

            # Move from this parent brand reply to the customer tweet it answered.
            cur_customer = norm_id(prow["customer_tweet_id"])

        return list(reversed(turns_rev))

    context_rows = []
    counts = []

    for i, r in pairs.iterrows():
        turns = get_previous_turns(r["customer_tweet_id"])
        counts.append(len(turns))

        out = {
            "customer_tweet_id": norm_id(r["customer_tweet_id"]),
            "brand_tweet_id": norm_id(r["brand_tweet_id"]),
            "customer_author_id": norm_id(r["customer_author_id"]),
            "customer_text": clean_text(r["customer_text"]),
            "brand_response": clean_text(r["brand_response"]),
            "customer_created_at": clean_text(r["customer_created_at"]),
            "brand_created_at": clean_text(r["brand_created_at"]),
            "customer_response_tweet_id": norm_id(r["customer_response_tweet_id"]),
            "brand_in_response_to": norm_id(r["brand_in_response_to"]),
            "customer_in_response_to": norm_id(r["customer_in_response_to"]),
        }

        for n in range(1, MAX_PREVIOUS_TURNS + 1):
            if len(turns) >= n:
                t = turns[-n]
                out[f"prev_{n}_customer"] = t["customer_text"]
                out[f"prev_{n}_brand"] = t["brand_response"]
                out[f"prev_{n}_customer_tweet_id"] = t["customer_tweet_id"]
                out[f"prev_{n}_brand_tweet_id"] = t["brand_tweet_id"]
            else:
                out[f"prev_{n}_customer"] = ""
                out[f"prev_{n}_brand"] = ""
                out[f"prev_{n}_customer_tweet_id"] = ""
                out[f"prev_{n}_brand_tweet_id"] = ""

        out["previous_turn_count"] = len(turns)
        context_rows.append(out)

    result = pd.DataFrame(context_rows)

    # Sanity checks.
    assert len(result) == len(pairs)
    assert (result["previous_turn_count"] >= 0).all()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, encoding="utf-8")

    print(f"Saved: {OUT}")
    print(f"Rows: {len(result):,}")
    print(f"Rows with true previous thread context: {(result.previous_turn_count > 0).sum():,}")
    print(f"Mean previous turns: {result.previous_turn_count.mean():.2f}")
    print("Previous-turn distribution:")
    print(result["previous_turn_count"].value_counts().sort_index().to_string())

    example = result[result["previous_turn_count"] > 0].head(1)
    if len(example):
        r = example.iloc[0]
        print("\nExample true thread:")
        for n in range(1, MAX_PREVIOUS_TURNS + 1):
            if r[f"prev_{n}_customer"]:
                print("-" * 60)
                print("Customer:", r[f"prev_{n}_customer"])
                print("AmazonHelp:", r[f"prev_{n}_brand"])
        print("-" * 60)
        print("CURRENT CUSTOMER:", r["customer_text"])


if __name__ == "__main__":
    main()
