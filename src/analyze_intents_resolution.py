import pandas as pd
import re
from pathlib import Path
from collections import Counter

INPUT = "data/processed/amazonhelp_contextual.csv"
OUTPUT_DIR = Path("data/processed/intent_resolution_analysis")


# ---------------------------------------------------------
# TEXT UTILITIES
# ---------------------------------------------------------

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_any(text, keywords):
    text = text.lower()
    return any(k in text for k in keywords)


# ---------------------------------------------------------
# INTENT RULES
# ---------------------------------------------------------
# These are NOT the final labels.
# They are discovery rules used to quantify recurring themes.

INTENT_RULES = {
    "delivery_not_received": [
        "not delivered",
        "wasn't delivered",
        "was not delivered",
        "didn't arrive",
        "did not arrive",
        "never arrived",
        "haven't received",
        "have not received",
        "not received",
        "don't have my",
        "dont have my",
        "missing package",
        "missing parcel",
        "parcel missing",
        "package missing",
        "says delivered",
        "marked delivered",
        "tracking says delivered",
    ],

    "delivery_delay": [
        "delivery delayed",
        "delivery delay",
        "late delivery",
        "delivery is late",
        "overdue",
        "over a week late",
        "days late",
        "still waiting",
        "waiting for delivery",
        "where is my package",
        "where's my package",
        "where is my parcel",
        "where's my parcel",
        "courier",
        "delivery date",
    ],

    "shipping_speed": [
        "shipping",
        "ship",
        "two day",
        "2 day",
        "one day",
        "next day",
        "expedited",
        "prime shipping",
        "shipping date",
    ],

    "order_status": [
        "order status",
        "where is my order",
        "order hasn't",
        "order hasnt",
        "order still",
        "track my order",
        "tracking",
        "expected delivery",
        "delivery estimate",
    ],

    "order_change_cancel": [
        "cancel my order",
        "cancel order",
        "cancelled my order",
        "change my order",
        "change order",
        "modify order",
        "wrong item",
        "wrong address",
    ],

    "refund_return": [
        "refund",
        "refunded",
        "return",
        "returned",
        "money back",
        "replacement",
        "replace",
        "exchange",
        "damaged",
        "broken",
        "arrived damaged",
    ],

    "payment_billing": [
        "payment",
        "charged",
        "charge",
        "billing",
        "credit card",
        "debit card",
        "card",
        "payment failed",
        "double charged",
        "charged twice",
        "amazon pay",
    ],

    "account_access": [
        "account",
        "password",
        "login",
        "log in",
        "sign in",
        "locked out",
        "close my account",
        "delete my account",
        "account closed",
        "account access",
    ],

    "product_device_issue": [
        "fire tv",
        "firetv",
        "echo",
        "alexa",
        "kindle",
        "prime video",
        "video error",
        "playback",
        "not working",
        "doesn't work",
        "doesnt work",
        "won't work",
        "wont work",
        "not supported",
        "troubleshooting",
        "device",
    ],

    "customer_service_escalation": [
        "customer service",
        "support",
        "phone",
        "call me",
        "call back",
        "callback",
        "chat",
        "speak to someone",
        "talk to someone",
        "human",
        "agent",
        "escalate",
    ],
}


# ---------------------------------------------------------
# RESOLUTION / ACTION RULES
# ---------------------------------------------------------

ACTION_RULES = {
    "self_service_help": [
        "help page",
        "help pages",
        "troubleshooting",
        "troubleshoot",
        "instructions",
        "article",
        "guide",
        "refer to",
        "check out",
        "visit",
        "link",
    ],

    "phone_or_chat": [
        "phone or chat",
        "phone",
        "chat",
        "customer service",
        "real time support",
        "live support",
        "contact us",
        "reach us",
    ],

    "callback": [
        "call back",
        "callback",
        "request a call",
        "request a callback",
    ],

    "delivery_investigation": [
        "take a closer look",
        "look into this",
        "look into your delivery",
        "investigate",
        "delivery",
        "order details",
        "check your order",
    ],

    "refund_or_replacement": [
        "refund",
        "replacement",
        "replace",
        "return",
        "available options",
    ],

    "information_or_explanation": [
        "information",
        "explain",
        "feedback",
        "supported",
        "available",
        "receive an email",
        "estimated delivery",
    ],

    "acknowledgement_or_closure": [
        "thank you",
        "thanks",
        "you're welcome",
        "you are welcome",
        "glad",
        "please let us know",
        "have a great",
        "schönen abend",
        "thanks for",
    ],

    "privacy_or_account_restriction": [
        "personal information",
        "unable to affect your account",
        "cannot affect your account",
        "unable to view",
        "unable to access",
        "via twitter",
    ],
}


# ---------------------------------------------------------
# CLASSIFIERS
# ---------------------------------------------------------

def classify_by_rules(text, rules):
    text = clean_text(text).lower()

    scores = {}

    for label, keywords in rules.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            scores[label] = score

    if not scores:
        return "other"

    # highest keyword match wins
    return max(scores, key=scores.get)


def classify_intent(row):
    # Context is deliberately included.
    combined = (
        str(row["conversation_context"]) + " " +
        str(row["current_customer_message"])
    )

    text = clean_text(combined)

    # More specific intents should win over generic ones.

    # Delivery not received is more specific than generic delivery.
    if contains_any(text, INTENT_RULES["delivery_not_received"]):
        return "delivery_not_received"

    if contains_any(text, INTENT_RULES["refund_return"]):
        return "refund_return"

    if contains_any(text, INTENT_RULES["account_access"]):
        return "account_access"

    if contains_any(text, INTENT_RULES["payment_billing"]):
        return "payment_billing"

    if contains_any(text, INTENT_RULES["product_device_issue"]):
        return "product_device_issue"

    if contains_any(text, INTENT_RULES["order_change_cancel"]):
        return "order_change_cancel"

    if contains_any(text, INTENT_RULES["delivery_delay"]):
        return "delivery_delay"

    if contains_any(text, INTENT_RULES["order_status"]):
        return "order_status"

    if contains_any(text, INTENT_RULES["shipping_speed"]):
        return "shipping_speed"

    if contains_any(text, INTENT_RULES["customer_service_escalation"]):
        return "customer_service_escalation"

    return "other"


def classify_action(text):
    return classify_by_rules(text, ACTION_RULES)


# ---------------------------------------------------------
# ESCALATION DETECTION
# ---------------------------------------------------------

ESCALATION_SIGNALS = {
    "explicit_support_request": [
        "call me",
        "call back",
        "callback",
        "phone",
        "chat",
        "speak to someone",
        "talk to someone",
        "human",
        "agent",
    ],

    "account_action_required": [
        "close my account",
        "delete my account",
        "account closed",
        "account access",
        "password",
        "locked out",
    ],

    "failed_self_service": [
        "didn't help",
        "did not help",
        "doesn't help",
        "doesnt help",
        "not helpful",
        "page is useless",
        "nothing helped",
        "still doesn't work",
        "still doesnt work",
        "still not working",
        "already tried",
    ],

    "missing_delivery": [
        "not delivered",
        "wasn't delivered",
        "was not delivered",
        "never arrived",
        "not received",
        "haven't received",
        "says delivered",
        "marked delivered",
    ],

    "high_frustration": [
        "wtf",
        "pissed",
        "furious",
        "ridiculous",
        "useless",
        "crazy",
        "sort it out",
        "terrible",
        "worst",
    ],
}


def escalation_signals(row):
    text = clean_text(
        str(row["conversation_context"]) + " " +
        str(row["current_customer_message"])
    ).lower()

    found = []

    for category, keywords in ESCALATION_SIGNALS.items():
        if any(k in text for k in keywords):
            found.append(category)

    return found


# ---------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("AMAZONHELP INTENT + RESOLUTION ANALYSIS")
    print("=" * 100)

    df = pd.read_csv(INPUT)

    print(f"Loaded rows: {len(df):,}")

    # -----------------------------------------------------
    # Intent classification
    # -----------------------------------------------------

    print("\nClassifying intents...")

    df["candidate_intent"] = df.apply(classify_intent, axis=1)

    intent_counts = (
        df["candidate_intent"]
        .value_counts()
        .rename_axis("candidate_intent")
        .reset_index(name="count")
    )

    intent_counts["percentage"] = (
        intent_counts["count"] / len(df) * 100
    ).round(2)

    print("\nCANDIDATE INTENT FREQUENCIES")
    print("-" * 100)
    print(intent_counts.to_string(index=False))

    intent_counts.to_csv(
        OUTPUT_DIR / "intent_frequencies.csv",
        index=False
    )

    # -----------------------------------------------------
    # Historical resolution/action
    # -----------------------------------------------------

    print("\nClassifying historical actions...")

    df["historical_action"] = df[
        "historical_brand_response"
    ].apply(classify_action)

    action_counts = (
        df["historical_action"]
        .value_counts()
        .rename_axis("historical_action")
        .reset_index(name="count")
    )

    action_counts["percentage"] = (
        action_counts["count"] / len(df) * 100
    ).round(2)

    print("\nRESOLUTION / ACTION FREQUENCIES")
    print("-" * 100)
    print(action_counts.to_string(index=False))

    action_counts.to_csv(
        OUTPUT_DIR / "resolution_frequencies.csv",
        index=False
    )

    # -----------------------------------------------------
    # Intent -> action mapping
    # -----------------------------------------------------

    mapping = pd.crosstab(
        df["candidate_intent"],
        df["historical_action"],
        normalize="index"
    ) * 100

    mapping = mapping.round(2)

    print("\nINTENT → HISTORICAL ACTION (%)")
    print("-" * 100)
    print(mapping.to_string())

    mapping.to_csv(
        OUTPUT_DIR / "intent_to_action_mapping.csv"
    )

    # -----------------------------------------------------
    # Examples for every intent
    # -----------------------------------------------------

    print("\nEXAMPLES BY INTENT")
    print("-" * 100)

    examples = []

    for intent in intent_counts["candidate_intent"]:

        subset = df[
            df["candidate_intent"] == intent
        ].head(10)

        print(f"\n### {intent}")

        for _, row in subset.iterrows():

            print(
                f"\nCustomer: "
                f"{row['current_customer_message'][:300]}"
            )

            print(
                f"AmazonHelp: "
                f"{row['historical_brand_response'][:300]}"
            )

            examples.append({
                "candidate_intent": intent,
                "customer_message":
                    row["current_customer_message"],
                "historical_response":
                    row["historical_brand_response"],
                "conversation_context":
                    row["conversation_context"],
            })

    pd.DataFrame(examples).to_csv(
        OUTPUT_DIR / "intent_examples.csv",
        index=False
    )

    # -----------------------------------------------------
    # Escalation analysis
    # -----------------------------------------------------

    print("\nDetecting escalation signals...")

    df["escalation_signals"] = df.apply(
        escalation_signals,
        axis=1
    )

    df["potential_escalation"] = (
        df["escalation_signals"].apply(len) > 0
    )

    escalation_df = df[
        df["potential_escalation"]
    ].copy()

    escalation_df["escalation_signals"] = (
        escalation_df["escalation_signals"]
        .apply(lambda x: ", ".join(x))
    )

    print(
        f"Potential escalation cases: "
        f"{len(escalation_df):,} "
        f"({len(escalation_df) / len(df) * 100:.2f}%)"
    )

    escalation_counts = Counter()

    for signals in df["escalation_signals"]:
        escalation_counts.update(signals)

    escalation_summary = pd.DataFrame(
        escalation_counts.items(),
        columns=["signal", "count"]
    ).sort_values(
        "count",
        ascending=False
    )

    print("\nESCALATION SIGNALS")
    print("-" * 100)
    print(escalation_summary.to_string(index=False))

    escalation_summary.to_csv(
        OUTPUT_DIR / "escalation_signals.csv",
        index=False
    )

    escalation_df[
        [
            "candidate_intent",
            "current_customer_message",
            "historical_brand_response",
            "conversation_context",
            "escalation_signals",
        ]
    ].head(500).to_csv(
        OUTPUT_DIR / "potential_escalation_examples.csv",
        index=False
    )

    # -----------------------------------------------------
    # Ambiguous/context dependent cases
    # -----------------------------------------------------

    print("\nFinding ambiguous/context-dependent cases...")

    ambiguous_patterns = [
        "don't have it",
        "dont have it",
        "still don't have it",
        "still dont have it",
        "nothing",
        "this",
        "that",
        "help",
        "please help",
        "thanks",
        "thank you",
        "okay",
        "ok",
        "what now",
        "now what",
        "still waiting",
        "not working",
    ]

    def is_ambiguous(row):

        current = clean_text(
            row["current_customer_message"]
        ).lower()

        if len(current.split()) <= 5:
            return True

        if any(
            pattern in current
            for pattern in ambiguous_patterns
        ):
            return True

        return False

    df["potentially_ambiguous"] = df.apply(
        is_ambiguous,
        axis=1
    )

    ambiguous_df = df[
        df["potentially_ambiguous"]
    ].copy()

    print(
        f"Potentially ambiguous/context-dependent: "
        f"{len(ambiguous_df):,} "
        f"({len(ambiguous_df) / len(df) * 100:.2f}%)"
    )

    ambiguous_df[
        [
            "candidate_intent",
            "current_customer_message",
            "conversation_context",
            "historical_brand_response",
        ]
    ].head(500).to_csv(
        OUTPUT_DIR / "ambiguous_examples.csv",
        index=False
    )

    # -----------------------------------------------------
    # Intent quality / context comparison
    # -----------------------------------------------------

    context_stats = (
        df.groupby(
            ["context_turn_count", "candidate_intent"]
        )
        .size()
        .reset_index(name="count")
    )

    context_stats.to_csv(
        OUTPUT_DIR / "intent_by_context_length.csv",
        index=False
    )

    # -----------------------------------------------------
    # Generate human-readable report
    # -----------------------------------------------------

    report_path = OUTPUT_DIR / "analysis_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:

        f.write(
            "AMAZONHELP INTENT + RESOLUTION ANALYSIS\n"
        )
        f.write("=" * 100 + "\n\n")

        f.write(
            f"Dataset size: {len(df):,}\n"
        )

        f.write(
            f"Unique customers: "
            f"{df['customer_author_id'].nunique():,}\n\n"
        )

        f.write(
            "CANDIDATE INTENT FREQUENCIES\n"
        )
        f.write("-" * 100 + "\n")
        f.write(
            intent_counts.to_string(index=False)
        )
        f.write("\n\n")

        f.write(
            "RESOLUTION / ACTION FREQUENCIES\n"
        )
        f.write("-" * 100 + "\n")
        f.write(
            action_counts.to_string(index=False)
        )
        f.write("\n\n")

        f.write(
            "INTENT → HISTORICAL ACTION (%)\n"
        )
        f.write("-" * 100 + "\n")
        f.write(
            mapping.to_string()
        )
        f.write("\n\n")

        f.write(
            "ESCALATION SIGNALS\n"
        )
        f.write("-" * 100 + "\n")
        f.write(
            escalation_summary.to_string(index=False)
        )
        f.write("\n\n")

        f.write(
            "AMBIGUOUS / CONTEXT-DEPENDENT CASES\n"
        )
        f.write("-" * 100 + "\n")
        f.write(
            f"{len(ambiguous_df):,} cases "
            f"({len(ambiguous_df) / len(df) * 100:.2f}%)\n"
        )

    # -----------------------------------------------------

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)

    print(f"\nAll outputs saved to:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.iterdir()):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()