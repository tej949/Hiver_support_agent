import pandas as pd
from pathlib import Path
import argparse


def analyze_brands(csv_path, chunksize=100_000):
    """
    Identify likely brand/support accounts from the Customer Support on Twitter dataset.

    In this dataset:
    - inbound=True  -> customer tweet
    - inbound=False -> support/brand reply

    A brand account is therefore better identified by having many outbound
    replies rather than requiring both inbound and outbound tweets.
    """

    outbound_counts = {}
    inbound_counts = {}
    total_counts = {}

    print(f"Reading: {csv_path}")
    print("This may take a few minutes for the full dataset...\n")

    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        low_memory=False
    ):
        # Normalize inbound column because CSV parsing can vary
        chunk["inbound"] = chunk["inbound"].astype(str).str.lower().eq("true")

        # Total tweets by author
        total = chunk["author_id"].value_counts()
        for author, count in total.items():
            total_counts[author] = total_counts.get(author, 0) + int(count)

        # Customer tweets
        inbound = chunk[chunk["inbound"]]
        counts = inbound["author_id"].value_counts()

        for author, count in counts.items():
            inbound_counts[author] = inbound_counts.get(author, 0) + int(count)

        # Brand/support tweets
        outbound = chunk[~chunk["inbound"]]
        counts = outbound["author_id"].value_counts()

        for author, count in counts.items():
            outbound_counts[author] = outbound_counts.get(author, 0) + int(count)

    # Build candidate list
    rows = []

    for author, outbound in outbound_counts.items():
        inbound = inbound_counts.get(author, 0)
        total = total_counts.get(author, 0)

        # Brand accounts should have substantial outbound activity.
        # We intentionally don't require inbound tweets.
        if outbound >= 500:
            rows.append({
                "brand_author_id": author,
                "total_tweets": total,
                "inbound_tweets": inbound,
                "outbound_tweets": outbound,
                "outbound_ratio": round(
                    outbound / total, 3
                ) if total else 0
            })

    result = pd.DataFrame(rows)

    if result.empty:
        print("No candidate support accounts found.")
        return

    # Strongest likely support accounts first
    result = result.sort_values(
        ["outbound_tweets", "total_tweets"],
        ascending=False
    )

    print("\nTOP CANDIDATE SUPPORT ACCOUNTS")
    print("=" * 100)
    print(result.head(50).to_string(index=False))

    output = Path("data/brand_analysis.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output, index=False)

    print(f"\nSaved full analysis to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find candidate customer-support accounts."
    )

    parser.add_argument(
        "csv",
        help="Path to twcs.csv"
    )

    args = parser.parse_args()

    analyze_brands(args.csv)