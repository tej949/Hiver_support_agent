from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

CONVERSATIONS_PATH = Path(
    "data/processed/amazonhelp_conversations.csv"
)

TRAIN_PATH = Path(
    "data/processed/splits/train_silver.csv"
)

VAL_PATH = Path(
    "data/processed/splits/validation_silver.csv"
)

OUTPUT_DIR = Path(
    "data/processed/baseline_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = (
    OUTPUT_DIR / "baseline_context_results.json"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR / "baseline_context_predictions.csv"
)

CONFUSION_PATH = (
    OUTPUT_DIR / "baseline_context_confusion_matrix.csv"
)


RANDOM_STATE = 42

INTENTS = [
    "delivery",
    "order_status_tracking",
    "order_change_cancel",
    "return_refund_replacement",
    "payment_billing",
    "account_access",
    "product_device_issue",
    "shipping_delivery_options",
    "customer_service_complaint",
    "information_general",
]


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------

def build_model():

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=100_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

def load_data():

    if not CONVERSATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Conversation file not found: "
            f"{CONVERSATIONS_PATH}"
        )

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training split not found: {TRAIN_PATH}"
        )

    if not VAL_PATH.exists():
        raise FileNotFoundError(
            f"Validation split not found: {VAL_PATH}"
        )

    conversations = pd.read_csv(
        CONVERSATIONS_PATH
    )

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VAL_PATH)

    required_conversation_columns = {
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_author_id",
        "customer_text",
        "brand_response",
        "customer_created_at",
        "brand_created_at",
        "customer_response_tweet_id",
        "brand_in_response_to",
    }

    missing = (
        required_conversation_columns
        - set(conversations.columns)
    )

    if missing:
        raise ValueError(
            "Conversation CSV is missing columns: "
            f"{sorted(missing)}"
        )

    return conversations, train, validation


# -------------------------------------------------------------------
# Normalize tweet IDs
# -------------------------------------------------------------------

def normalize_id(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


# -------------------------------------------------------------------
# Build conversation context
# -------------------------------------------------------------------

def build_context_map(conversations):

    df = conversations.copy()

    # Normalize IDs.
    for col in [
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_response_tweet_id",
        "brand_in_response_to",
    ]:
        df[col] = df[col].apply(normalize_id)

    # Make sure dates sort correctly.
    df["customer_created_at_dt"] = pd.to_datetime(
        df["customer_created_at"],
        errors="coerce",
        utc=True,
    )

    df["brand_created_at_dt"] = pd.to_datetime(
        df["brand_created_at"],
        errors="coerce",
        utc=True,
    )

    # Sort chronologically.
    df = df.sort_values(
        [
            "customer_author_id",
            "customer_created_at_dt",
            "brand_created_at_dt",
        ]
    )

    # ---------------------------------------------------------------
    # For each customer message, collect previous turns from the
    # same customer.
    #
    # We intentionally limit the context to the previous 3 customer
    # turns and their Amazon responses.
    # ---------------------------------------------------------------

    context_map = {}

    histories = {}

    for _, row in df.iterrows():

        customer_id = str(
            row["customer_author_id"]
        )

        current_id = normalize_id(
            row["customer_tweet_id"]
        )

        if current_id is None:
            continue

        history = histories.get(
            customer_id,
            []
        )

        previous_turns = history[-3:]

        context_parts = []

        for turn in previous_turns:

            context_parts.append(
                "Customer: "
                + str(turn["customer_text"])
            )

            context_parts.append(
                "AmazonHelp: "
                + str(turn["brand_response"])
            )

        current_message = str(
            row["customer_text"]
        )

        if context_parts:

            combined = (
                "\n".join(context_parts)
                + "\nCustomer: "
                + current_message
            )

        else:

            combined = current_message

        context_map[current_id] = combined

        # Add current turn to history for subsequent messages.
        history.append(
            {
                "customer_text": current_message,
                "brand_response": str(
                    row["brand_response"]
                ),
            }
        )

        histories[customer_id] = history

    return context_map


# -------------------------------------------------------------------
# Attach context to split
# -------------------------------------------------------------------

def attach_context(
    split_df,
    context_map,
):

    result = split_df.copy()

    result["customer_tweet_id"] = (
        result["customer_tweet_id"]
        .apply(normalize_id)
    )

    result["context_text"] = (
        result["customer_tweet_id"]
        .map(context_map)
    )

    # If a context lookup fails, fall back to current message.
    result["context_text"] = (
        result["context_text"]
        .fillna(result["customer_text"])
    )

    return result


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "BASELINE 2 — CONTEXT-AWARE "
        "TF-IDF + LOGISTIC REGRESSION"
    )
    print("=" * 70)

    conversations, train, validation = load_data()

    print(
        f"\nFull conversation pairs: "
        f"{len(conversations):,}"
    )

    print(
        f"Training examples:   "
        f"{len(train):,}"
    )

    print(
        f"Validation examples: "
        f"{len(validation):,}"
    )

    # ---------------------------------------------------------------
    # Reconstruct context
    # ---------------------------------------------------------------

    print(
        "\nReconstructing previous conversation turns..."
    )

    context_map = build_context_map(
        conversations
    )

    print(
        f"Context entries created: "
        f"{len(context_map):,}"
    )

    # ---------------------------------------------------------------
    # Attach context
    # ---------------------------------------------------------------

    train_context = attach_context(
        train,
        context_map,
    )

    val_context = attach_context(
        validation,
        context_map,
    )

    # ---------------------------------------------------------------
    # Measure how much context we actually obtained.
    # ---------------------------------------------------------------

    train_has_context = (
        train_context["context_text"]
        != train_context["customer_text"]
    )

    val_has_context = (
        val_context["context_text"]
        != val_context["customer_text"]
    )

    print(
        "\nExamples containing previous context:"
    )

    print(
        f"  Train: "
        f"{train_has_context.sum():,} / "
        f"{len(train_context):,}"
    )

    print(
        f"  Validation: "
        f"{val_has_context.sum():,} / "
        f"{len(val_context):,}"
    )

    # ---------------------------------------------------------------
    # Prepare features
    # ---------------------------------------------------------------

    X_train = (
        train_context["context_text"]
        .fillna("")
        .astype(str)
    )

    y_train = (
        train_context["silver_intent"]
        .astype(str)
    )

    X_val = (
        val_context["context_text"]
        .fillna("")
        .astype(str)
    )

    y_val = (
        val_context["silver_intent"]
        .astype(str)
    )

    # ---------------------------------------------------------------
    # Show real context example.
    # ---------------------------------------------------------------

    context_examples = train_context[
        train_has_context
    ]

    print(
        "\nExample with conversation context:"
    )
    print("-" * 70)

    if len(context_examples) > 0:

        example = context_examples.iloc[0]

        print(
            example["context_text"][:2000]
        )

    else:

        print(
            "WARNING: No previous context found."
        )

    print("-" * 70)

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    print(
        "\nTraining context-aware model..."
    )

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # ---------------------------------------------------------------
    # Predict
    # ---------------------------------------------------------------

    print(
        "\nGenerating validation predictions..."
    )

    predictions = model.predict(X_val)

    probabilities = model.predict_proba(
        X_val
    )

    confidence = probabilities.max(
        axis=1
    )

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        predictions,
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_val,
        predictions,
        labels=INTENTS,
        output_dict=True,
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy:    {accuracy:.4f}"
    )

    print(
        f"Macro F1:    {macro_f1:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    print(
        "\nPer-intent performance:"
    )

    print(
        classification_report(
            y_val,
            predictions,
            labels=INTENTS,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------------

    confidence_series = pd.Series(
        confidence
    )

    print("Confidence statistics:")

    print(
        f"  Mean:   "
        f"{confidence_series.mean():.4f}"
    )

    print(
        f"  Median: "
        f"{confidence_series.median():.4f}"
    )

    print(
        f"  Min:    "
        f"{confidence_series.min():.4f}"
    )

    print(
        f"  Max:    "
        f"{confidence_series.max():.4f}"
    )

    # ---------------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------------

    output = validation.copy()

    output["predicted_intent"] = (
        predictions
    )

    output["prediction_confidence"] = (
        confidence
    )

    output["correct"] = (
        output["predicted_intent"]
        == output["silver_intent"]
    )

    output.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print(
        "\nPredictions saved to:"
    )

    print(
        f"  {PREDICTIONS_PATH}"
    )

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=INTENTS,
    )

    confusion_df = pd.DataFrame(
        cm,
        index=INTENTS,
        columns=INTENTS,
    )

    confusion_df.to_csv(
        CONFUSION_PATH
    )

    print(
        "Confusion matrix saved to:"
    )

    print(
        f"  {CONFUSION_PATH}"
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    results = {
        "model": (
            "Context-aware TF-IDF + "
            "Logistic Regression"
        ),
        "task": "intent_classification",
        "input": (
            "previous_customer_and_brand_turns"
            "+current_customer_message"
        ),
        "context_turns": 3,
        "training_data": {
            "type": "silver_labels",
            "path": str(TRAIN_PATH),
            "rows": int(len(train)),
        },
        "validation_data": {
            "type": "silver_labels",
            "path": str(VAL_PATH),
            "rows": int(len(validation)),
        },
        "context_coverage": {
            "train_with_context": int(
                train_has_context.sum()
            ),
            "train_total": int(
                len(train_context)
            ),
            "validation_with_context": int(
                val_has_context.sum()
            ),
            "validation_total": int(
                len(val_context)
            ),
        },
        "golden_set_used": False,
        "random_state": RANDOM_STATE,
        "metrics": {
            "accuracy": float(
                accuracy
            ),
            "macro_f1": float(
                macro_f1
            ),
            "weighted_f1": float(
                weighted_f1
            ),
        },
        "confidence": {
            "mean": float(
                confidence_series.mean()
            ),
            "median": float(
                confidence_series.median()
            ),
            "min": float(
                confidence_series.min()
            ),
            "max": float(
                confidence_series.max()
            ),
        },
        "classification_report": report,
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        "\nMetrics saved to:"
    )

    print(
        f"  {RESULTS_PATH}"
    )

    # ---------------------------------------------------------------
    # Sample errors
    # ---------------------------------------------------------------

    mistakes = output[
        ~output["correct"]
    ].copy()

    print("\n" + "=" * 70)
    print("SAMPLE ERRORS")
    print("=" * 70)

    if len(mistakes) > 0:

        sample = mistakes.sample(
            min(10, len(mistakes)),
            random_state=RANDOM_STATE,
        )

        for _, row in sample.iterrows():

            print("\nCustomer:")

            print(
                str(
                    row["customer_text"]
                )[:500]
            )

            print(
                f"Actual:     "
                f"{row['silver_intent']}"
            )

            print(
                f"Predicted:  "
                f"{row['predicted_intent']}"
            )

            print(
                f"Confidence: "
                f"{row['prediction_confidence']:.4f}"
            )

    else:

        print("No errors found.")

    print("\n" + "=" * 70)
    print("BASELINE 2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()