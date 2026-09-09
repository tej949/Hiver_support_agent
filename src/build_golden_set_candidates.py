import pandas as pd
import numpy as np
import re
from pathlib import Path


INPUT = "data/processed/amazonhelp_contextual.csv"
OUTPUT_DIR = Path("data/processed/golden_set_candidates")

RANDOM_STATE = 42

# Number of candidates to retain per bucket before creating
# the final review pool.
PER_BUCKET = 60


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_any(text, patterns):
    text = text.lower()

    return any(
        pattern.lower() in text
        for pattern in patterns
    )


# ============================================================
# INTENT DISCOVERY RULES
# ============================================================
#
# IMPORTANT:
# These are candidate-generation rules ONLY.
# They are NOT ground truth.
#
# Human labelers will assign the final intent.
# ============================================================

INTENT_RULES = {

    "delivery": [
        "not delivered",
        "wasn't delivered",
        "was not delivered",
        "didn't arrive",
        "did not arrive",
        "never arrived",
        "not received",
        "haven't received",
        "have not received",
        "still hasn't arrived",
        "still hasnt arrived",
        "still haven't received",
        "still havent received",
        "package",
        "parcel",
        "delivery",
        "delivered",
        "courier",
        "carrier",
        "driver",
        "out for delivery",
        "failed delivery",
        "wrong address",
        "mis-delivered",
        "missing delivery",
    ],

    "order": [
        "order status",
        "where is my order",
        "track my order",
        "tracking",
        "order number",
        "order #",
        "order hasn't",
        "order hasnt",
        "order still",
        "ordered",
        "preorder",
        "pre-order",
        "expected delivery",
        "delivery estimate",
    ],

    "refund": [
        "refund",
        "refunded",
        "money back",
        "return",
        "returned",
        "replacement",
        "replace",
        "exchange",
        "damaged",
        "broken",
        "wrong item",
        "incorrect item",
        "arrived damaged",
    ],

    "payment": [
        "payment",
        "charged",
        "charge",
        "billing",
        "credit card",
        "debit card",
        "card",
        "payment failed",
        "charged twice",
        "double charged",
        "unknown charge",
        "false charge",
        "prime membership",
        "prime charge",
        "gift card",
    ],

    "account": [
        "account",
        "password",
        "login",
        "log in",
        "sign in",
        "locked out",
        "close my account",
        "delete my account",
        "account access",
        "email address",
        "deregister",
        "security",
    ],

    "product": [
        "fire tv",
        "firetv",
        "echo",
        "alexa",
        "kindle",
        "prime video",
        "video",
        "playback",
        "device",
        "tablet",
        "not working",
        "doesn't work",
        "doesnt work",
        "won't work",
        "wont work",
        "error",
        "troubleshoot",
        "troubleshooting",
        "sd card",
    ],

    "shipping": [
        "shipping",
        "ship",
        "shipped",
        "two day",
        "2 day",
        "one day",
        "next day",
        "same day",
        "expedited",
        "prime shipping",
        "shipping date",
        "launch day",
    ],

    "complaint": [
        "customer service",
        "support",
        "pissed",
        "furious",
        "terrible",
        "ridiculous",
        "useless",
        "worst",
        "frustrated",
        "frustration",
        "angry",
        "complaint",
        "complaints",
        "fed up",
        "done with amazon",
        "lost me as a customer",
        "speak to someone",
        "talk to someone",
        "human",
        "agent",
        "call me",
        "callback",
        "call back",
        "phone",
        "chat",
        "escalate",
    ],

    "information": [
        "how do i",
        "how can i",
        "can i",
        "is it possible",
        "what is",
        "what does",
        "where can i",
        "when do",
        "when will",
        "why",
        "supported",
        "available",
        "information",
        "question",
    ],

    "cancellation": [
        "cancel my order",
        "cancel order",
        "cancelled my order",
        "cancelled order",
        "cancel my",
        "cancellation",
        "reorder",
        "re-order",
        "change my order",
        "modify order",
        "wrong address",
    ],
}


# ============================================================
# ACTION / RESOLUTION RULES
# ============================================================

ACTION_RULES = {

    "self_service": [
        "help page",
        "help pages",
        "troubleshooting",
        "troubleshoot",
        "instructions",
        "article",
        "guide",
        "check out",
        "visit",
        "see here",
        "learn more",
    ],

    "investigation": [
        "look into",
        "take a closer look",
        "look deeper",
        "investigate",
        "check into",
        "review",
        "check your order",
        "check the tracking",
        "provide your details",
        "provide some details",
    ],

    "phone_chat": [
        "phone or chat",
        "phone/chat",
        "contact us by phone",
        "contact us via phone",
        "reach us by phone",
        "reach us",
        "real-time support",
        "live support",
        "customer service",
    ],

    "callback": [
        "call back",
        "callback",
        "request a call",
        "request a callback",
        "give you a call",
    ],

    "refund_replacement": [
        "refund",
        "replacement",
        "replace",
        "return",
        "available options",
    ],

    "information": [
        "you will receive",
        "you'll receive",
        "you can",
        "you may",
        "is supported",
        "please check",
        "information",
        "feedback",
    ],

    "acknowledgement": [
        "thank you",
        "thanks",
        "you're welcome",
        "you are welcome",
        "glad",
        "happy to help",
        "have a great",
        "please let us know",
        "thanks for",
    ],

    "privacy_restriction": [
        "personal information",
        "personal info",
        "account details",
        "unable to view",
        "unable to affect",
        "via twitter",
        "public",
        "security",
    ],
}


# ============================================================
# ESCALATION SIGNALS
# ============================================================

ESCALATION_RULES = {

    "explicit_human_request": [
        "call me",
        "call back",
        "callback",
        "speak to someone",
        "talk to someone",
        "human",
        "agent",
        "phone",
        "chat",
    ],

    "failed_support": [
        "didn't help",
        "did not help",
        "doesn't help",
        "doesnt help",
        "nothing helped",
        "not helpful",
        "page is useless",
        "already tried",
        "tried everything",
        "still doesn't work",
        "still doesnt work",
        "still not working",
    ],

    "account_action": [
        "close my account",
        "delete my account",
        "locked out",
        "can't access my account",
        "cannot access my account",
        "unable to access my account",
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
        "missing package",
        "missing parcel",
    ],

    "high_frustration": [
        "wtf",
        "pissed",
        "furious",
        "ridiculous",
        "useless",
        "crazy",
        "terrible",
        "worst",
        "angry",
        "frustrated",
        "sort it out",
        "done with amazon",
        "lost me as a customer",
    ],
}


# ============================================================
# CANDIDATE INTENT
# ============================================================

def intent_scores(text):

    scores = {}

    for intent, patterns in INTENT_RULES.items():

        matches = [
            p for p in patterns
            if p.lower() in text.lower()
        ]

        scores[intent] = len(matches)

    return scores


def candidate_intent(row):

    current = clean_text(
        row["current_customer_message"]
    )

    context = clean_text(
        row["conversation_context"]
    )

    # Current message receives higher weight.
    combined = current + " " + context

    scores = intent_scores(combined)

    best_intent = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_intent]

    # No useful signal
    if best_score == 0:
        return "information"

    return best_intent


# ============================================================
# CANDIDATE ACTION
# ============================================================

def candidate_action(response):

    response = clean_text(response)

    scores = {}

    for action, patterns in ACTION_RULES.items():

        scores[action] = sum(
            1
            for p in patterns
            if p.lower() in response.lower()
        )

    best = max(
        scores,
        key=scores.get
    )

    if scores[best] == 0:
        return "other"

    return best


# ============================================================
# ESCALATION SIGNALS
# ============================================================

def get_escalation_signals(row):

    text = (
        clean_text(row["current_customer_message"])
        + " "
        + clean_text(row["conversation_context"])
    ).lower()

    signals = []

    for signal, patterns in ESCALATION_RULES.items():

        if any(
            p.lower() in text
            for p in patterns
        ):
            signals.append(signal)

    return signals


# ============================================================
# AMBIGUITY
# ============================================================

def ambiguity_score(row):

    current = clean_text(
        row["current_customer_message"]
    )

    context = clean_text(
        row["conversation_context"]
    )

    words = current.split()

    score = 0

    # Very short messages
    if len(words) <= 3:
        score += 2

    elif len(words) <= 6:
        score += 1

    # Pronouns/deictic references
    ambiguous_terms = [
        "this",
        "that",
        "it",
        "there",
        "here",
        "still",
        "nothing",
        "same",
        "again",
        "okay",
        "ok",
        "yes",
        "no",
        "sadly",
    ]

    if any(
        term in current.lower().split()
        for term in ambiguous_terms
    ):
        score += 2

    # Context is likely important when context has
    # multiple turns.
    if len(context) > 150:
        score += 1

    return score


# ============================================================
# STRATIFIED SAMPLING
# ============================================================

def sample_candidates(df):

    rng = np.random.default_rng(RANDOM_STATE)

    samples = []

    # --------------------------------------------------------
    # 1. Intent buckets
    # --------------------------------------------------------

    for intent in INTENT_RULES.keys():

        subset = df[
            df["candidate_intent"] == intent
        ].copy()

        if len(subset) == 0:
            continue

        # Prefer examples with context.
        subset["selection_score"] = (
            subset["context_turn_count"] * 2
            + subset["ambiguity_score"]
            + subset["escalation_signals"].apply(len)
        )

        # Keep a mixture of difficult and ordinary examples.
        subset = subset.sort_values(
            "selection_score",
            ascending=False
        )

        top_n = min(
            PER_BUCKET,
            len(subset)
        )

        # Half difficult, half random.
        difficult_n = top_n // 2

        difficult = subset.head(
            difficult_n
        )

        remaining = subset.iloc[
            difficult_n:
        ]

        random_n = top_n - len(difficult)

        if len(remaining) > random_n:
            random_part = remaining.sample(
                random_n,
                random_state=RANDOM_STATE
            )
        else:
            random_part = remaining

        samples.append(
            pd.concat(
                [difficult, random_part]
            )
        )

    # --------------------------------------------------------
    # 2. Ambiguous bucket
    # --------------------------------------------------------

    ambiguous = df[
        df["ambiguity_score"] >= 2
    ].copy()

    if len(ambiguous) > 0:

        ambiguous = ambiguous.sort_values(
            [
                "ambiguity_score",
                "context_turn_count"
            ],
            ascending=False
        )

        samples.append(
            ambiguous.head(PER_BUCKET)
        )

    # --------------------------------------------------------
    # 3. Escalation bucket
    # --------------------------------------------------------

    escalation = df[
        df["escalation_signals"].apply(len) > 0
    ].copy()

    if len(escalation) > 0:

        escalation["escalation_strength"] = (
            escalation["escalation_signals"]
            .apply(len)
        )

        escalation = escalation.sort_values(
            "escalation_strength",
            ascending=False
        )

        samples.append(
            escalation.head(PER_BUCKET)
        )

    result = pd.concat(
        samples,
        ignore_index=True
    )

    # Remove duplicate tweet pairs
    result = result.drop_duplicates(
        subset=[
            "customer_tweet_id",
            "brand_tweet_id"
        ]
    )

    # Randomize final order
    result = result.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 100)
    print("GOLDEN SET CANDIDATE SAMPLER")
    print("=" * 100)

    print(f"\nLoading: {INPUT}")

    df = pd.read_csv(
        INPUT,
        low_memory=False
    )

    print(
        f"Loaded: {len(df):,} rows"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "customer_tweet_id",
        "customer_author_id",
        "current_customer_message",
        "historical_brand_response",
        "conversation_context",
        "context_turn_count",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df["current_customer_message"] = (
        df["current_customer_message"]
        .fillna("")
        .astype(str)
    )

    df["conversation_context"] = (
        df["conversation_context"]
        .fillna("")
        .astype(str)
    )

    df["historical_brand_response"] = (
        df["historical_brand_response"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Candidate annotations
    # --------------------------------------------------------

    print("\nGenerating candidate intents...")

    df["candidate_intent"] = df.apply(
        candidate_intent,
        axis=1
    )

    print("\nGenerating candidate actions...")

    df["candidate_action"] = (
        df["historical_brand_response"]
        .apply(candidate_action)
    )

    print("\nGenerating escalation signals...")

    df["escalation_signals"] = df.apply(
        get_escalation_signals,
        axis=1
    )

    df["escalation_signals"] = (
        df["escalation_signals"]
        .apply(
            lambda x: "|".join(x)
        )
    )

    print("\nCalculating ambiguity...")

    df["ambiguity_score"] = df.apply(
        ambiguity_score,
        axis=1
    )

    # --------------------------------------------------------
    # Output columns
    # --------------------------------------------------------

    output_columns = [
        "customer_tweet_id",
        "customer_author_id",
        "brand_tweet_id",
        "current_customer_message",
        "conversation_context",
        "historical_brand_response",
        "candidate_intent",
        "candidate_action",
        "escalation_signals",
        "context_turn_count",
        "ambiguity_score",
    ]

    # --------------------------------------------------------
    # Individual intent candidate files
    # --------------------------------------------------------

    print("\nCreating individual intent candidate files...")

    filename_map = {
        "delivery": "delivery_candidates.csv",
        "order": "order_candidates.csv",
        "refund": "refund_candidates.csv",
        "payment": "payment_candidates.csv",
        "account": "account_candidates.csv",
        "product": "product_candidates.csv",
        "shipping": "shipping_candidates.csv",
        "complaint": "complaint_candidates.csv",
        "information": "information_candidates.csv",
        "cancellation": "cancellation_candidates.csv",
    }

    for intent, filename in filename_map.items():

        subset = df[
            df["candidate_intent"] == intent
        ].copy()

        # Rank by useful human-review characteristics.
        subset["review_priority"] = (
            subset["ambiguity_score"]
            + subset["escalation_signals"]
                .str.count(r"\|")
            + subset["context_turn_count"]
        )

        subset = subset.sort_values(
            "review_priority",
            ascending=False
        )

        subset = subset.drop(
            columns=["review_priority"]
        )

        subset[output_columns].head(
            PER_BUCKET
        ).to_csv(
            OUTPUT_DIR / filename,
            index=False
        )

        print(
            f"  {intent:15s}: "
            f"{len(subset):,} available"
        )

    # --------------------------------------------------------
    # Ambiguous candidates
    # --------------------------------------------------------

    ambiguous = df[
        df["ambiguity_score"] >= 2
    ].copy()

    ambiguous = ambiguous.sort_values(
        [
            "ambiguity_score",
            "context_turn_count"
        ],
        ascending=False
    )

    ambiguous[output_columns].head(
        100
    ).to_csv(
        OUTPUT_DIR / "ambiguous_candidates.csv",
        index=False
    )

    # --------------------------------------------------------
    # Escalation candidates
    # --------------------------------------------------------

    escalation = df[
        df["escalation_signals"].str.len() > 0
    ].copy()

    escalation["signal_count"] = (
        escalation["escalation_signals"]
        .str.count(r"\|") + 1
    )

    escalation = escalation.sort_values(
        "signal_count",
        ascending=False
    )

    escalation[output_columns].head(
        100
    ).to_csv(
        OUTPUT_DIR / "escalation_candidates.csv",
        index=False
    )

    # --------------------------------------------------------
    # Final review pool
    # --------------------------------------------------------

    print("\nCreating stratified Golden Set review pool...")

    review_pool = sample_candidates(df)

    # Keep exactly the fields needed for labeling.
    review_columns = [
        "customer_tweet_id",
        "customer_author_id",
        "brand_tweet_id",
        "current_customer_message",
        "conversation_context",
        "historical_brand_response",
        "candidate_intent",
        "candidate_action",
        "escalation_signals",
        "context_turn_count",
        "ambiguity_score",
    ]

    review_pool = review_pool[
        review_columns
    ]

    review_pool.to_csv(
        OUTPUT_DIR / "golden_set_sampling.csv",
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("CANDIDATE DISTRIBUTION")
    print("=" * 100)

    print(
        df["candidate_intent"]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 100)
    print("REVIEW POOL")
    print("=" * 100)

    print(
        f"Total review candidates: "
        f"{len(review_pool):,}"
    )

    print("\nCandidate intent distribution:")

    print(
        review_pool["candidate_intent"]
        .value_counts()
        .to_string()
    )

    print("\nCandidate action distribution:")

    print(
        review_pool["candidate_action"]
        .value_counts()
        .to_string()
    )

    print("\nEscalation candidates:")

    print(
        (
            review_pool["escalation_signals"]
            .str.len() > 0
        ).sum()
    )

    print("\nAmbiguous candidates:")

    print(
        (
            review_pool["ambiguity_score"] >= 2
        ).sum()
    )

    # --------------------------------------------------------
    # Labeling template
    # --------------------------------------------------------

    labeling = review_pool.copy()

    labeling["gold_intent"] = ""
    labeling["gold_action"] = ""
    labeling["gold_handling"] = ""
    labeling["gold_reason"] = ""
    labeling["gold_evidence_sufficient"] = ""

    labeling.to_csv(
        OUTPUT_DIR / "golden_set_labeling_template.csv",
        index=False
    )

    print("\nGenerated:")
    print(
        "  golden_set_sampling.csv"
    )
    print(
        "  golden_set_labeling_template.csv"
    )

    print("\nSaved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()