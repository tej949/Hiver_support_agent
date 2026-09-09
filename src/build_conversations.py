import pandas as pd
from pathlib import Path
import argparse


def normalize_id(value):
    """Convert tweet IDs such as 272.0 into '272'."""
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def build_conversations(csv_path, brand, output_path, chunksize=100_000):

    print(f"Loading dataset: {csv_path}")
    print(f"Target brand: {brand}\n")

    # ---------------------------------------------------------
    # 1. Load dataset
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

    # ---------------------------------------------------------
    # 2. Normalize columns
    # ---------------------------------------------------------

    df["inbound"] = (
        df["inbound"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    df["tweet_id"] = df["tweet_id"].apply(normalize_id)

    df["in_response_to_tweet_id"] = (
        df["in_response_to_tweet_id"]
        .apply(normalize_id)
    )

    df["response_tweet_id"] = (
        df["response_tweet_id"]
        .apply(normalize_id)
    )

    # ---------------------------------------------------------
    # 3. Build tweet lookup
    # ---------------------------------------------------------

    tweet_lookup = df.set_index("tweet_id")

    # ---------------------------------------------------------
    # 4. Get AmazonHelp replies
    # ---------------------------------------------------------

    brand_replies = df[
        (df["author_id"] == brand) &
        (~df["inbound"])
    ].copy()

    print(f"Brand replies found: {len(brand_replies):,}")

    # ---------------------------------------------------------
    # 5. Connect replies to customer tweets
    # ---------------------------------------------------------

    conversations = []

    matched = 0
    missing_parent = 0
    brand_parent = 0

    for _, reply in brand_replies.iterrows():

        parent_id = reply["in_response_to_tweet_id"]

        if parent_id is None:
            continue

        if parent_id not in tweet_lookup.index:
            missing_parent += 1
            continue

        customer = tweet_lookup.loc[parent_id]

        # Don't treat AmazonHelp → AmazonHelp as a customer case
        if customer["author_id"] == brand:
            brand_parent += 1
            continue

        matched += 1

        conversations.append({
            "customer_tweet_id": parent_id,
            "brand_tweet_id": reply["tweet_id"],

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

    # ---------------------------------------------------------
    # 6. Create dataframe safely
    # ---------------------------------------------------------

    columns = [
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_author_id",
        "customer_text",
        "brand_response",
        "customer_created_at",
        "brand_created_at",
        "customer_response_tweet_id",
        "brand_in_response_to"
    ]

    conversations_df = pd.DataFrame(
        conversations,
        columns=columns
    )

    conversations_df = conversations_df.drop_duplicates(
        subset=["brand_tweet_id"]
    )

    # ---------------------------------------------------------
    # 7. Save
    # ---------------------------------------------------------

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conversations_df.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------------------------------
    # 8. Statistics
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("CONVERSATION DATASET")
    print("=" * 80)

    print(f"Customer → Brand pairs: {len(conversations_df):,}")

    if len(conversations_df) > 0:
        print(
            f"Unique customers: "
            f"{conversations_df['customer_author_id'].nunique():,}"
        )

    print(f"Matched replies:       {matched:,}")
    print(f"Missing parent tweet:  {missing_parent:,}")
    print(f"Brand-parent tweets:   {brand_parent:,}")

    print(f"\nSaved to: {output_path}")

    # ---------------------------------------------------------
    # 9. Show examples
    # ---------------------------------------------------------

    if len(conversations_df) > 0:

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
        default="data/processed/amazonhelp_conversations.csv"
    )

    args = parser.parse_args()

    build_conversations(
        args.csv,
        args.brand,
        args.out
    )