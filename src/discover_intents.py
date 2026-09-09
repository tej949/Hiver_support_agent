import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean customer tweets while preserving words useful for
    intent discovery.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove HTML entities
    text = re.sub(r"&\w+;", " ", text)

    # Remove RT marker
    text = re.sub(r"\brt\b", " ", text)

    # Keep apostrophes inside words
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# BASIC QUALITY FILTER
# ============================================================

def is_useful_message(text):
    """
    Remove messages that are too short to provide meaningful
    evidence for intent discovery.
    """

    if not text:
        return False

    words = text.split()

    # Very short messages are usually greetings/thanks/noise
    if len(words) < 3:
        return False

    # Remove messages consisting almost entirely of punctuation
    if len("".join(words)) < 5:
        return False

    return True


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):

    print(f"Loading: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = [
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_text",
        "brand_response"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(f"Total customer → brand pairs: {len(df):,}")

    # Clean customer messages
    df["clean_text"] = df["customer_text"].apply(clean_text)

    before = len(df)

    df = df[
        df["clean_text"].apply(is_useful_message)
    ].copy()

    removed = before - len(df)

    print(f"Removed short/noisy messages: {removed:,}")
    print(f"Messages remaining: {len(df):,}")

    # Remove duplicate customer messages
    df = df.drop_duplicates(
        subset=["clean_text"]
    ).reset_index(drop=True)

    print(
        f"After duplicate removal: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# FREQUENT WORDS / PHRASES
# ============================================================

def show_common_terms(df):

    print("\n")
    print("=" * 90)
    print("COMMON CUSTOMER ISSUE TERMS")
    print("=" * 90)

    vectorizer = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS),
        ngram_range=(1, 2),
        min_df=20,
        max_df=0.80,
        max_features=5000,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(
        df["clean_text"]
    )

    terms = vectorizer.get_feature_names_out()

    # Mean TF-IDF across documents
    scores = np.asarray(
        X.mean(axis=0)
    ).ravel()

    ranking = np.argsort(scores)[::-1]

    print("\nTop terms / phrases by TF-IDF importance:\n")

    for i in ranking[:100]:
        print(
            f"{terms[i]:40s} "
            f"{scores[i]:.5f}"
        )

    return vectorizer


# ============================================================
# CLUSTERING
# ============================================================

def cluster_for_k(df, k):

    print("\n")
    print("=" * 90)
    print(f"K-MEANS INTENT DISCOVERY — K={k}")
    print("=" * 90)

    vectorizer = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS),
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.90,
        max_features=15000,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(
        df["clean_text"]
    )

    print(
        f"TF-IDF matrix: "
        f"{X.shape[0]:,} documents × "
        f"{X.shape[1]:,} features"
    )

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=300
    )

    labels = model.fit_predict(X)

    df_result = df.copy()
    df_result["cluster"] = labels

    # Silhouette score on a sample to avoid excessive runtime
    sample_size = min(10000, X.shape[0])

    if X.shape[0] > sample_size:

        rng = np.random.RandomState(42)

        indices = rng.choice(
            X.shape[0],
            size=sample_size,
            replace=False
        )

        score = silhouette_score(
            X[indices],
            labels[indices]
        )

    else:

        score = silhouette_score(
            X,
            labels
        )

    print(
        f"\nSilhouette score: "
        f"{score:.4f}"
    )

    # --------------------------------------------------------
    # Cluster details
    # --------------------------------------------------------

    feature_names = vectorizer.get_feature_names_out()

    cluster_sizes = (
        df_result["cluster"]
        .value_counts()
        .sort_index()
    )

    print("\nCLUSTER SIZES")
    print("-" * 90)

    for cluster, size in cluster_sizes.items():

        percentage = (
            size / len(df_result) * 100
        )

        print(
            f"Cluster {cluster:2d}: "
            f"{size:7,} messages "
            f"({percentage:5.2f}%)"
        )

    # --------------------------------------------------------
    # Top terms per cluster
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("TOP TERMS + REPRESENTATIVE EXAMPLES")
    print("=" * 90)

    centers = model.cluster_centers_

    all_cluster_rows = []

    for cluster in range(k):

        print("\n")
        print("#" * 90)
        print(
            f"CLUSTER {cluster} "
            f"({cluster_sizes.get(cluster, 0):,} messages)"
        )
        print("#" * 90)

        # Top TF-IDF features
        top_indices = centers[cluster].argsort()[::-1][:15]

        top_terms = [
            feature_names[i]
            for i in top_indices
        ]

        print("\nTop terms:")
        print(", ".join(top_terms))

        # ----------------------------------------------------
        # Find representative messages
        # ----------------------------------------------------

        cluster_indices = np.where(
            labels == cluster
        )[0]

        cluster_vectors = X[cluster_indices]

        # Distance to centroid
                # ----------------------------------------------------
        # Distance to centroid
        # ----------------------------------------------------
        #
        # Use Euclidean distance correctly with sparse TF-IDF.
        # We calculate:
        #
        # ||x-c||² = ||x||² + ||c||² - 2x.c
        #

        centroid = centers[cluster]

        x_squared = np.asarray(
            cluster_vectors.multiply(
                cluster_vectors
            ).sum(axis=1)
        ).ravel()

        centroid_squared = np.sum(
            centroid ** 2
        )

        dot_product = (
            cluster_vectors @ centroid
        )

        dot_product = np.asarray(
            dot_product
        ).ravel()

        distances = (
            x_squared
            + centroid_squared
            - 2 * dot_product
        )

        # Numerical precision can occasionally produce
        # tiny negative values.
        distances = np.maximum(
            distances,
            0
        )

        representative_order = np.argsort(
            distances
        )[:10]

        print("\nRepresentative customer messages:")

        for rank, position in enumerate(
            representative_order,
            start=1
        ):

            original_index = cluster_indices[position]

            text = df_result.iloc[
                original_index
            ]["customer_text"]

            cleaned = df_result.iloc[
                original_index
            ]["clean_text"]

            print(
                f"\n{rank}. {text}"
            )

            all_cluster_rows.append({
                "k": k,
                "cluster": cluster,
                "cluster_size": int(
                    cluster_sizes.get(cluster, 0)
                ),
                "top_terms": ", ".join(top_terms),
                "representative_rank": rank,
                "customer_text": text,
                "clean_text": cleaned
            })

    return (
        df_result,
        score,
        all_cluster_rows
    )


# ============================================================
# COMPARE K VALUES
# ============================================================

def compare_clusters(df, values):

    print("\n")
    print("=" * 90)
    print("CHOOSING THE INTENT GRANULARITY")
    print("=" * 90)

    results = []
    representative_rows = []

    for k in values:

        clustered, score, rows = cluster_for_k(
            df,
            k
        )

        results.append({
            "k": k,
            "silhouette_score": round(score, 4),
            "smallest_cluster": int(
                clustered["cluster"]
                .value_counts().min()
            ),
            "largest_cluster": int(
                clustered["cluster"]
                .value_counts().max()
            )
        })

        representative_rows.extend(rows)

    result_df = pd.DataFrame(results)

    print("\n")
    print(result_df.to_string(index=False))

    return result_df, pd.DataFrame(
        representative_rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Discover candidate customer-support "
            "intents from AmazonHelp conversations."
        )
    )

    parser.add_argument(
        "input",
        help=(
            "Path to "
            "data/processed/amazonhelp_conversations.csv"
        )
    )

    parser.add_argument(
        "--output",
        default="data/processed/intent_discovery",
        help="Output directory"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data(args.input)

    # --------------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------------

    cleaned_path = (
        output_dir /
        "cleaned_customer_messages.csv"
    )

    df.to_csv(
        cleaned_path,
        index=False
    )

    print(
        f"\nSaved cleaned messages to: "
        f"{cleaned_path}"
    )

    # --------------------------------------------------------
    # Common terms
    # --------------------------------------------------------

    show_common_terms(df)

    # --------------------------------------------------------
    # Compare K = 8, 10, 12
    # --------------------------------------------------------

    comparison, representatives = compare_clusters(
        df,
        [8, 10, 12]
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    comparison_path = (
        output_dir /
        "cluster_comparison.csv"
    )

    representatives_path = (
        output_dir /
        "cluster_representatives.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    representatives.to_csv(
        representatives_path,
        index=False
    )

    print("\n")
    print("=" * 90)
    print("FILES CREATED")
    print("=" * 90)

    print(
        f"Cleaned data:       {cleaned_path}"
    )

    print(
        f"Cluster comparison: {comparison_path}"
    )

    print(
        f"Representatives:    {representatives_path}"
    )

    print("\nIntent discovery complete.")


if __name__ == "__main__":
    main()