import pandas as pd
from pathlib import Path
import argparse


def explore_brand(csv_path, brand, chunksize=100_000):
    print(f"Analyzing brand: {brand}")
    print(f"Reading: {csv_path}\n")

    brand_tweets = []

    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        low_memory=False
    ):
        chunk["inbound"] = (
            chunk["inbound"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

        matches = chunk[chunk["author_id"] == brand]

        if len(matches):
            brand_tweets.append(matches)

    df = pd.concat(brand_tweets, ignore_index=True)

    print("=" * 80)
    print("BRAND SUMMARY")
    print("=" * 80)

    print(f"Brand:              {brand}")
    print(f"Total brand tweets: {len(df):,}")
    print(f"Brand replies:      {(~df['inbound']).sum():,}")
    print(f"Brand inbound:      {df['inbound'].sum():,}")

    print("\nSample brand replies:")
    print("-" * 80)

    replies = df[~df["inbound"]]

    for _, row in replies.head(30).iterrows():
        print(f"\nTweet ID: {row['tweet_id']}")
        print(f"Response to: {row['in_response_to_tweet_id']}")
        print(f"Text: {row['text']}")

    # Save AmazonHelp tweets for easier inspection
    output = Path(f"data/{brand}_tweets.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv",
        help="Path to twcs.csv"
    )

    parser.add_argument(
        "--brand",
        required=True,
        help="Brand author ID, e.g. AmazonHelp"
    )

    args = parser.parse_args()

    explore_brand(
        args.csv,
        args.brand
    )