import pandas as pd
from pathlib import Path
import argparse


def build_conversations(csv_path, brand, output_path, chunksize=100_000):
    print(f"Loading dataset: {csv_path}")
    print(f"Target brand: {brand}\n")

    # ---------------------------------------------------------
    # 1. Load the complete tweet dataset
    # ---------------------------------------------------------

    parts = []

    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        low_memory=False
    ):
        parts.append(chunk)

    df = pd.concat(parts, ignore_index=True)

    print(f"Total tweets loaded: {len(df):,}")

    # Normalize inbound
    df["inbound"] = (
        df["inbound"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    # ---------------------------------------------------------
    # 2. Create lookup by tweet ID
    # ---------------------------------------------------------

    df["tweet_id"] = df["tweet_id"].astype(str)

    tweet_lookup = df.set_index("tweet_id")

    # ---------------------------------------------------------
    # 3. Find all brand responses
    # ---------------------------------------------------------

    brand_replies = df[
        (df["author_id"] == brand) &
        (~df["inbound"])
    ].copy()

    print(f"Brand replies found: {len(brand_replies):,}")

    # ---------------------------------------------------------
    # 4. Connect every brand reply to the tweet it answers
    # ---------------------------------------------------------

    conversations = []

    for _, reply in brand_replies.iterrows():

        parent_id = reply["in_response_to_tweet_id"]

        if pd.isna(parent_id):
            continue

        parent_id = str(parent_id)

        if parent_id not in tweet_lookup.index:
            continue

        customer = tweet_lookup.loc[parent_id]

        # We only want replies to customer tweets
        if customer["author_id"] == brand:
            continue

        conversations.append({
            "customer_tweet_id": parent_id,
            "brand_tweet_id": str(reply["tweet_id"]),

            "customer_author_id": customer["author_id"],

            "customer_text": str(customer["text"]),
            "brand_response": str(reply["text"]),

            "customer_created_at": customer["created_at"],
            "brand_created_at": reply["created_at"],

            "customer_response_tweet_id":
                customer["response_tweet_id"],

            "brand_in_response_to":
                reply["in_response_to_tweet_id"]
        })

    conversations_df = pd.DataFrame(conversations)

    # ---------------------------------------------------------
    # 5. Remove duplicates
    # ---------------------------------------------------------

    conversations_df = conversations_df.drop_duplicates(
        subset=["brand_tweet_id"]
    )

    # ---------------------------------------------------------
    # 6. Save
    # ---------------------------------------------------------

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conversations_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("CONVERSATION DATASET")
    print("=" * 80)

    print(
        f"Customer → Brand pairs: "
        f"{len(conversations_df):,}"
    )

    print(
        f"Unique customers: "
        f"{conversations_df['customer_author_id'].nunique():,}"
    )

    print(
        f"Saved to: {output_path}"
    )

    # ---------------------------------------------------------
    # 7. Show examples
    # ---------------------------------------------------------

    print("\nSAMPLE CONVERSATIONS")
    print("-" * 80)

    for _, row in conversations_df.head(20).iterrows():

        print("\nCUSTOMER:")
        print(row["customer_text"])

        print("\nAMAZONHELP:")
        print(row["brand_response"])

        print("-" * 80)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv",
        help="Path to twcs.csv"
    )

    parser.add_argument(
        "--brand",
        required=True,
        help="Brand author ID"
    )

    parser.add_argument(
        "--out",
        default="data/processed/amazonhelp_conversations.csv",
        help="Output CSV path"
    )

    args = parser.parse_args()

    build_conversations(
        args.csv,
        args.brand,
        args.out
    )