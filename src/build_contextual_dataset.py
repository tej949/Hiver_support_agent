import pandas as pd
from pathlib import Path
import argparse


# ============================================================
# HELPERS
# ============================================================

def normalize_id(value):
    """
    Normalize tweet IDs such as:
        272.0 -> "272"
        272   -> "272"
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def clean_text(text):
    """Basic whitespace normalization."""

    if pd.isna(text):
        return ""

    return " ".join(str(text).split())


# ============================================================
# LOAD DATA
# ============================================================

def load_tweets(csv_path, chunksize=100_000):

    print(f"Loading: {csv_path}")

    parts = []

    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        low_memory=False
    ):
        parts.append(chunk)

    df = pd.concat(
        parts,
        ignore_index=True
    )

    print(f"Total tweets: {len(df):,}")

    # Normalize IDs
    df["tweet_id"] = df["tweet_id"].apply(
        normalize_id
    )

    df["response_tweet_id"] = (
        df["response_tweet_id"]
        .apply(normalize_id)
    )

    df["in_response_to_tweet_id"] = (
        df["in_response_to_tweet_id"]
        .apply(normalize_id)
    )

    # Normalize inbound flag
    df["inbound"] = (
        df["inbound"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    return df


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_contextual_dataset(
    csv_path,
    brand,
    output_path,
    context_turns=3
):

    df = load_tweets(csv_path)

    # --------------------------------------------------------
    # Tweet lookup
    # --------------------------------------------------------

    tweet_lookup = df.set_index(
        "tweet_id",
        drop=False
    )

    # --------------------------------------------------------
    # Find all customer tweets that AmazonHelp responds to
    # --------------------------------------------------------

    brand_replies = df[
        (df["author_id"] == brand) &
        (~df["inbound"])
    ].copy()

    print(
        f"Brand replies: "
        f"{len(brand_replies):,}"
    )

    records = []

    matched = 0
    missing_customer = 0

    # --------------------------------------------------------
    # Process every AmazonHelp reply
    # --------------------------------------------------------

    for _, brand_reply in brand_replies.iterrows():

        customer_id = (
            brand_reply["in_response_to_tweet_id"]
        )

        if customer_id is None:
            continue

        if customer_id not in tweet_lookup.index:
            missing_customer += 1
            continue

        current_customer = tweet_lookup.loc[
            customer_id
        ]

        # Skip if the parent is another AmazonHelp tweet
        if current_customer["author_id"] == brand:
            continue

        matched += 1

        # ----------------------------------------------------
        # Build previous conversation turns
        # ----------------------------------------------------

        context_messages = []

        current_id = customer_id

        # Walk backwards through the conversation.
        #
        # response_tweet_id tells us what tweet(s) this tweet
        # responds to.
        #
        # We use the first response ID when multiple IDs exist.
        #

        visited = set()

        for _ in range(context_turns):

            if current_id is None:
                break

            if current_id in visited:
                break

            visited.add(current_id)

            if current_id not in tweet_lookup.index:
                break

            tweet = tweet_lookup.loc[current_id]

            text = clean_text(
                tweet["text"]
            )

            if text:

                if tweet["author_id"] == brand:
                    speaker = "AmazonHelp"
                else:
                    speaker = "Customer"

                context_messages.append(
                    f"{speaker}: {text}"
                )

            previous_id = (
                tweet["in_response_to_tweet_id"]
            )

            if previous_id is None:
                break

            current_id = previous_id

        # ----------------------------------------------------
        # Reverse because we walked backwards
        # ----------------------------------------------------

        context_messages.reverse()

        # ----------------------------------------------------
        # Make sure current customer message is included
        # ----------------------------------------------------

        current_customer_text = clean_text(
            current_customer["text"]
        )

        # ----------------------------------------------------
        # Historical AmazonHelp response
        # ----------------------------------------------------

        historical_response = clean_text(
            brand_reply["text"]
        )

        # ----------------------------------------------------
        # Create record
        # ----------------------------------------------------

        records.append({

            "customer_tweet_id":
                customer_id,

            "brand_tweet_id":
                brand_reply["tweet_id"],

            "customer_author_id":
                current_customer["author_id"],

            "current_customer_message":
                current_customer_text,

            "historical_brand_response":
                historical_response,

            "conversation_context":
                "\n".join(context_messages),

            "context_turn_count":
                len(context_messages),

            "customer_created_at":
                current_customer["created_at"],

            "brand_created_at":
                brand_reply["created_at"]
        })

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    columns = [
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_author_id",
        "current_customer_message",
        "historical_brand_response",
        "conversation_context",
        "context_turn_count",
        "customer_created_at",
        "brand_created_at"
    ]

    contextual_df = pd.DataFrame(
        records,
        columns=columns
    )

    # Remove exact duplicate examples
    contextual_df = contextual_df.drop_duplicates(
        subset=[
            "customer_tweet_id",
            "brand_tweet_id"
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    contextual_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("CONTEXTUAL DATASET")
    print("=" * 90)

    print(
        f"Customer → Brand examples: "
        f"{len(contextual_df):,}"
    )

    print(
        f"Unique customers: "
        f"{contextual_df['customer_author_id'].nunique():,}"
    )

    print(
        f"Matched conversations: "
        f"{matched:,}"
    )

    print(
        f"Missing customer tweets: "
        f"{missing_customer:,}"
    )

    print(
        f"Average context turns: "
        f"{contextual_df['context_turn_count'].mean():.2f}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    # --------------------------------------------------------
    # Context distribution
    # --------------------------------------------------------

    print("\nCONTEXT LENGTH DISTRIBUTION")
    print("-" * 90)

    distribution = (
        contextual_df[
            "context_turn_count"
        ]
        .value_counts()
        .sort_index()
    )

    for turns, count in distribution.items():

        print(
            f"{turns} turns: "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Representative examples
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("SAMPLE CONTEXTUAL CONVERSATIONS")
    print("=" * 90)

    for i, (_, row) in enumerate(
        contextual_df.head(20).iterrows(),
        start=1
    ):

        print(f"\nEXAMPLE {i}")
        print("-" * 90)

        print(
            "CONTEXT:"
        )

        print(
            row["conversation_context"]
        )

        print(
            "\nCURRENT CUSTOMER MESSAGE:"
        )

        print(
            row["current_customer_message"]
        )

        print(
            "\nHISTORICAL AMAZONHELP RESPONSE:"
        )

        print(
            row["historical_brand_response"]
        )

        print("-" * 90)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Build contextual customer-support "
            "examples from the Customer Support "
            "on Twitter dataset."
        )
    )

    parser.add_argument(
        "csv",
        help="Path to twcs.csv"
    )

    parser.add_argument(
        "--brand",
        required=True,
        help="Brand account, e.g. AmazonHelp"
    )

    parser.add_argument(
        "--out",
        default=(
            "data/processed/"
            "amazonhelp_contextual.csv"
        ),
        help="Output CSV path"
    )

    parser.add_argument(
        "--context-turns",
        type=int,
        default=3,
        help=(
            "Maximum number of conversation "
            "turns to include"
        )
    )

    args = parser.parse_args()

    build_contextual_dataset(
        args.csv,
        args.brand,
        args.out,
        args.context_turns
    )