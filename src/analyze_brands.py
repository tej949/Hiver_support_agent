import pandas as pd
from pathlib import Path
import argparse


def analyze_brands(csv_path, chunksize=100_000):
    counts = {}
    inbound_counts = {}
    outbound_counts = {}

    print(f"Reading: {csv_path}")
    print("This may take a few minutes for the full dataset...\n")

    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        low_memory=False
    ):
        # Count all tweets by author
        author_counts = chunk["author_id"].value_counts()

        for author, count in author_counts.items():
            counts[author] = counts.get(author, 0) + int(count)

        # Incoming customer tweets
        inbound = chunk[chunk["inbound"] == True]
        inbound_author_counts = inbound["author_id"].value_counts()

        for author, count in inbound_author_counts.items():
            inbound_counts[author] = inbound_counts.get(author, 0) + int(count)

        # Outgoing brand replies
        outbound = chunk[chunk["inbound"] == False]
        outbound_author_counts = outbound["author_id"].value_counts()

        for author, count in outbound_author_counts.items():
            outbound_counts[author] = outbound_counts.get(author, 0) + int(count)

    rows = []

    for author, total in counts.items():
        inbound = inbound_counts.get(author, 0)
        outbound = outbound_counts.get(author, 0)

        # A support brand should have substantial inbound AND outbound activity.
        if inbound >= 100 and outbound >= 100:
            rows.append({
                "brand": author,
                "total_tweets": total,
                "inbound_tweets": inbound,
                "outbound_tweets": outbound,
                "response_ratio": round(
                    outbound / inbound, 3
                ) if inbound else 0
            })

    result = pd.DataFrame(rows)

    if result.empty:
        print("No suitable brands found.")
        return

    result = result.sort_values(
        ["inbound_tweets", "outbound_tweets"],
        ascending=False
    )

    print("\nTOP SUPPORT BRANDS")
    print("=" * 80)

    print(
        result.head(30).to_string(index=False)
    )

    output = Path("data/brand_analysis.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print(f"\nSaved full analysis to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find candidate customer-support brands."
    )

    parser.add_argument(
        "csv",
        help="Path to twcs.csv"
    )

    args = parser.parse_args()

    analyze_brands(args.csv)