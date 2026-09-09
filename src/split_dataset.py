from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


INPUT = Path("data/processed/amazonhelp_conversations.csv")
OUTPUT_DIR = Path("data/processed/splits")

RANDOM_STATE = 42


def assign_silver_intent(text):
    """
    Weak/silver labeling only.
    These labels are used for training baselines, NOT treated as ground truth.
    """

    text = str(text).lower()

    # Order change / cancellation
    if any(x in text for x in [
        "cancel my order",
        "cancel order",
        "cancelled my order",
        "change my order",
        "wrong address",
        "change address",
        "shipping address",
        "cancel my",
    ]):
        return "order_change_cancel"

    # Returns / refunds / replacements
    if any(x in text for x in [
        "refund",
        "return",
        "replacement",
        "replace",
        "exchange",
        "damaged",
        "defective",
        "wrong item",
        "money back",
    ]):
        return "return_refund_replacement"

    # Payment / billing
    if any(x in text for x in [
        "cashback",
        "charged",
        "charge",
        "payment",
        "billing",
        "invoice",
        "emi",
        "price",
        "pricing",
        "amazon pay",
        "money deducted",
    ]):
        return "payment_billing"

    # Account access
    if any(x in text for x in [
        "can't login",
        "cannot login",
        "can't log in",
        "cannot log in",
        "locked account",
        "account locked",
        "account suspended",
        "account hacked",
        "hacked my account",
        "password",
        "verification code",
        "two step",
        "2 step",
        "sign in",
        "login",
    ]):
        return "account_access"

    # Product/device
    if any(x in text for x in [
        "kindle",
        "echo",
        "alexa",
        "fire tv",
        "firetv",
        "prime video",
        "remote",
        "device",
        "speaker",
        "tablet",
        "dvd",
        "not working",
        "stopped working",
    ]):
        return "product_device_issue"

    # Customer-service complaint
    if any(x in text for x in [
        "customer service",
        "customer care",
        "representative",
        "representatives",
        "support team",
        "worst service",
        "poor service",
        "bad service",
        "unprofessional",
        "unhelpful",
        "no response",
        "no reply",
        "not responding",
        "still no response",
        "complaint",
        "escalate",
    ]):
        return "customer_service_complaint"

    # Delivery failure
    if any(x in text for x in [
        "not delivered",
        "not received",
        "haven't received",
        "have not received",
        "never arrived",
        "didn't arrive",
        "did not arrive",
        "late delivery",
        "delivery is late",
        "delivery delayed",
        "delivered but",
        "marked delivered",
        "wrong address",
        "misdelivered",
        "missing package",
        "missing parcel",
    ]):
        return "delivery"

    # Order status / tracking
    if any(x in text for x in [
        "where is my order",
        "where's my order",
        "where is my package",
        "where's my package",
        "tracking",
        "track my order",
        "order status",
        "delivery status",
        "when will my order",
        "when will it arrive",
        "eta",
    ]):
        return "order_status_tracking"

    # Shipping options
    if any(x in text for x in [
        "next day",
        "one day delivery",
        "two day delivery",
        "shipping option",
        "shipping options",
        "expedited shipping",
        "shipping speed",
        "prime delivery",
        "amazon locker",
        "ship to",
        "shipping to",
        "delivery option",
        "delivery options",
    ]):
        return "shipping_delivery_options"

    # General information
    if any(x in text for x in [
        "how do i",
        "how can i",
        "can i",
        "is it possible",
        "what is",
        "when is",
        "available",
        "availability",
        "information",
    ]):
        return "information_general"

    return "unknown"


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT}. "
            "Run build_conversations.py first."
        )

    df = pd.read_csv(INPUT)

    print(f"Loaded conversations: {len(df):,}")

    # Make sure customer grouping exists
    if "customer_author_id" not in df.columns:
        raise ValueError(
            "customer_author_id column not found. "
            "Check amazonhelp_conversations.csv."
        )

    # Current customer message is what the classifier sees.
    df["silver_intent"] = df["customer_text"].apply(assign_silver_intent)

    # Remove unknown silver labels from supervised training.
    usable = df[df["silver_intent"] != "unknown"].copy()

    print(f"Usable silver-labeled rows: {len(usable):,}")
    print("\nSilver intent distribution:")
    print(usable["silver_intent"].value_counts())

    # ---------------------------------------------------------
    # First split: 85% development, 15% held-out development
    # ---------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=RANDOM_STATE
    )

    train_idx, val_idx = next(
        splitter.split(
            usable,
            groups=usable["customer_author_id"]
        )
    )

    train = usable.iloc[train_idx].copy()
    validation = usable.iloc[val_idx].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train_silver.csv"
    val_path = OUTPUT_DIR / "validation_silver.csv"

    train.to_csv(train_path, index=False)
    validation.to_csv(val_path, index=False)

    print("\nCreated:")
    print(f"  Train:      {len(train):,}")
    print(f"  Validation: {len(validation):,}")

    print("\nCustomer leakage check:")

    train_customers = set(train["customer_author_id"])
    val_customers = set(validation["customer_author_id"])

    overlap = train_customers & val_customers

    print(f"  Train customers:      {len(train_customers):,}")
    print(f"  Validation customers: {len(val_customers):,}")
    print(f"  Overlapping:          {len(overlap)}")

    assert len(overlap) == 0

    print("\nIMPORTANT:")
    print("The 200-row Golden Set remains untouched.")
    print("It will be used as the human-labeled evaluation set.")
    print("Silver labels are training labels only and are not ground truth.")

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()