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

TRAIN_PATH = Path("data/processed/splits/train_silver.csv")
VAL_PATH = Path("data/processed/splits/validation_silver.csv")

OUTPUT_DIR = Path("data/processed/baseline_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "baseline_tfidf_results.json"
PREDICTIONS_PATH = OUTPUT_DIR / "baseline_tfidf_predictions.csv"


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

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


def load_data():
    """Load silver-labeled training and validation data."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Validation file not found: {VAL_PATH}")

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VAL_PATH)

    required_columns = {"customer_text", "silver_intent"}

    for name, df in [
        ("train", train),
        ("validation", validation),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

    return train, validation


def build_model():
    """
    Baseline:
        customer message
             ↓
          TF-IDF
             ↓
       Logistic Regression
             ↓
           intent
    """

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


def main():

    print("=" * 70)
    print("BASELINE 1 — TF-IDF + LOGISTIC REGRESSION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    train, validation = load_data()

    print(f"\nTraining examples:   {len(train):,}")
    print(f"Validation examples: {len(validation):,}")

    X_train = train["customer_text"].fillna("").astype(str)
    y_train = train["silver_intent"].astype(str)

    X_val = validation["customer_text"].fillna("").astype(str)
    y_val = validation["silver_intent"].astype(str)

    print("\nTraining intent distribution:")
    print(y_train.value_counts())

    # ---------------------------------------------------------------
    # Build + train model
    # ---------------------------------------------------------------

    print("\nTraining TF-IDF + Logistic Regression...")

    model = build_model()
    model.fit(X_train, y_train)

    print("Training complete.")

    # ---------------------------------------------------------------
    # Predict
    # ---------------------------------------------------------------

    print("\nGenerating validation predictions...")

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)

    confidence = probabilities.max(axis=1)

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    accuracy = accuracy_score(y_val, predictions)

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

    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nPer-intent performance:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=INTENTS,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------------
    # Confidence statistics
    # ---------------------------------------------------------------

    print("Confidence statistics:")

    confidence_series = pd.Series(confidence)

    print(f"  Mean:   {confidence_series.mean():.4f}")
    print(f"  Median: {confidence_series.median():.4f}")
    print(f"  Min:    {confidence_series.min():.4f}")
    print(f"  Max:    {confidence_series.max():.4f}")

    # ---------------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------------

    output = validation.copy()

    output["predicted_intent"] = predictions
    output["prediction_confidence"] = confidence

    output["correct"] = (
        output["predicted_intent"] == output["silver_intent"]
    )

    output.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\nPredictions saved to:")
    print(f"  {PREDICTIONS_PATH}")

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

    confusion_path = OUTPUT_DIR / "baseline_tfidf_confusion_matrix.csv"

    confusion_df.to_csv(confusion_path)

    print(f"Confusion matrix saved to:")
    print(f"  {confusion_path}")

    # ---------------------------------------------------------------
    # Save machine-readable metrics
    # ---------------------------------------------------------------

    results = {
        "model": "TF-IDF + Logistic Regression",
        "task": "intent_classification",
        "input": "current_customer_message_only",
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
        "golden_set_used": False,
        "random_state": RANDOM_STATE,
        "metrics": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        },
        "confidence": {
            "mean": float(confidence_series.mean()),
            "median": float(confidence_series.median()),
            "min": float(confidence_series.min()),
            "max": float(confidence_series.max()),
        },
        "classification_report": report,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nMetrics saved to:")
    print(f"  {RESULTS_PATH}")

    # ---------------------------------------------------------------
    # Show a few mistakes
    # ---------------------------------------------------------------

    mistakes = output[~output["correct"]].copy()

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
            print(str(row["customer_text"])[:500])

            print(f"Actual:    {row['silver_intent']}")
            print(f"Predicted: {row['predicted_intent']}")
            print(
                f"Confidence: {row['prediction_confidence']:.4f}"
            )

    else:
        print("No errors found.")

    print("\n" + "=" * 70)
    print("BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()