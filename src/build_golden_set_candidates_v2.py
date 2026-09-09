from pathlib import Path
import pandas as pd
import numpy as np
import re

# ============================================================
# CONFIG
# ============================================================

INPUT = Path("data/processed/amazonhelp_contextual.csv")
OUTPUT_DIR = Path("data/processed/golden_v2")

SEED = 42
RECOMMENDED_SIZE = 200
BACKUP_SIZE = 600

rng = np.random.default_rng(SEED)

# Final candidate intent taxonomy
INTENTS = [
    "delivery",
    "order_status_tracking",
    "return_refund_replacement",
    "payment_billing",
    "account_access",
    "product_device_issue",
    "shipping_delivery_options",
    "customer_service_complaint",
    "information_general",
    "order_change_cancel",
]

# Main quotas: 170 examples
QUOTAS = {
    "delivery": 22,
    "order_status_tracking": 18,
    "return_refund_replacement": 22,
    "payment_billing": 18,
    "account_access": 15,
    "product_device_issue": 22,
    "shipping_delivery_options": 15,
    "customer_service_complaint": 18,
    "information_general": 12,
    "order_change_cancel": 8,
}

# Additional difficult examples
AMBIGUOUS_QUOTA = 15
ESCALATION_QUOTA = 15

# ============================================================
# TEXT NORMALIZATION
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\brt\b", " ", text)

    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# KEYWORD / PHRASE SIGNALS
#
# IMPORTANT:
# These are ONLY for sampling.
# They are NOT ground truth labels.
# ============================================================

INTENT_PATTERNS = {

    "delivery": [
        "where is my package",
        "where is my order",
        "order hasn't arrived",
        "order hasnt arrived",
        "package hasn't arrived",
        "package hasnt arrived",
        "not delivered",
        "wasn't delivered",
        "wasnt delivered",
        "never arrived",
        "still waiting for my order",
        "still waiting for my package",
        "missing package",
        "missing order",
        "marked delivered",
        "says delivered",
        "delivered but",
        "delivery delayed",
        "delivery delay",
        "late delivery",
        "delivery is late",
        "delivery hasn't",
        "delivery hasnt",
        "amazon logistics",
    ],

    "order_status_tracking": [
        "track my order",
        "track order",
        "tracking number",
        "tracking information",
        "tracking info",
        "order status",
        "status of my order",
        "where's my order",
        "wheres my order",
        "where is my order",
        "when will my order arrive",
        "when will it arrive",
        "eta",
        "estimated delivery",
        "delivery date",
        "shipping status",
    ],

    "return_refund_replacement": [
        "return my",
        "return an item",
        "return item",
        "want to return",
        "return this",
        "refund",
        "money back",
        "get my money back",
        "replacement",
        "replace my",
        "replace this",
        "exchange",
        "damaged item",
        "damaged package",
        "arrived damaged",
        "wrong item",
        "received wrong",
        "broken item",
        "item is broken",
        "defective",
        "missing item",
    ],

    "payment_billing": [
        "payment",
        "paid",
        "charge",
        "charged",
        "billing",
        "bill",
        "invoice",
        "credit card",
        "debit card",
        "card was charged",
        "charged twice",
        "double charged",
        "payment failed",
        "payment declined",
        "payment method",
        "unauthorized charge",
        "unknown charge",
        "subscription charge",
    ],

    "account_access": [
        "account",
        "login",
        "log in",
        "sign in",
        "signin",
        "password",
        "forgot password",
        "reset password",
        "can't access",
        "cant access",
        "unable to access",
        "locked out",
        "account locked",
        "close my account",
        "delete my account",
        "account closed",
        "verification",
        "verify my account",
    ],

    "product_device_issue": [
        "fire tv",
        "firetv",
        "kindle",
        "echo",
        "alexa",
        "prime video",
        "primevideo",
        "device",
        "remote",
        "screen",
        "won't turn on",
        "wont turn on",
        "not working",
        "doesn't work",
        "doesnt work",
        "not connecting",
        "can't connect",
        "cant connect",
        "error message",
        "error code",
        "playback",
        "streaming",
        "app isn't working",
        "app isnt working",
    ],

    "shipping_delivery_options": [
        "shipping options",
        "shipping option",
        "delivery options",
        "delivery option",
        "same day delivery",
        "one day delivery",
        "two day shipping",
        "next day delivery",
        "free shipping",
        "prime shipping",
        "shipping cost",
        "delivery cost",
        "deliver to",
        "international shipping",
        "ship to",
        "shipping speed",
        "faster shipping",
    ],

    "customer_service_complaint": [
        "customer service",
        "customer support",
        "support is",
        "terrible service",
        "bad service",
        "worst service",
        "poor service",
        "no one is helping",
        "nobody is helping",
        "no help",
        "been trying to contact",
        "can't get help",
        "cant get help",
        "speak to someone",
        "talk to someone",
        "human",
        "agent",
        "representative",
        "complaint",
        "frustrated",
        "angry",
        "ridiculous",
        "unacceptable",
    ],

    "information_general": [
        "how do i",
        "how can i",
        "what is",
        "what are",
        "can i",
        "is there",
        "do you have",
        "does amazon",
        "when does",
        "where can i",
        "information",
        "question",
        "wondering",
        "tell me",
    ],

    "order_change_cancel": [
        "cancel my order",
        "cancel order",
        "cancel this order",
        "change my order",
        "modify my order",
        "change order",
        "wrong address",
        "change address",
        "shipping address",
        "change quantity",
        "remove item",
        "add item to order",
        "ordered by mistake",
        "cancelled order",
        "cancelled my order",
    ],
}


# ============================================================
# ACTION SIGNALS
# ============================================================

ACTION_PATTERNS = {
    "self_service": [
        "click here",
        "go here",
        "visit",
        "follow this link",
        "help page",
        "troubleshooting",
        "try these steps",
        "try this",
        "restart",
        "reset",
    ],

    "investigation": [
        "look into",
        "investigate",
        "check with",
        "check the order",
        "delivery investigation",
        "look into this",
        "escalate the order",
        "amazon logistics",
    ],

    "phone_chat": [
        "call",
        "phone",
        "chat",
        "customer service",
        "contact us",
        "speak with",
        "talk to",
    ],

    "callback": [
        "call you",
        "callback",
        "call back",
        "phone number",
        "contact number",
    ],

    "refund_replacement": [
        "refund",
        "replacement",
        "replace",
        "return",
        "money back",
    ],

    "information": [
        "information",
        "details",
        "explain",
        "let me know",
        "here's how",
        "heres how",
    ],

    "acknowledgement": [
        "you're welcome",
        "youre welcome",
        "glad",
        "happy to help",
        "sorry",
        "apologies",
        "thanks for",
    ],

    "privacy_restriction": [
        "can't access your account",
        "cant access your account",
        "unable to view",
        "cannot view",
        "privacy",
        "personal information",
        "security reasons",
    ],
}


# ============================================================
# ESCALATION SIGNALS
# ============================================================

ESCALATION_PATTERNS = {
    "explicit_human_request": [
        "speak to a human",
        "speak to someone",
        "talk to someone",
        "real person",
        "human agent",
        "customer service agent",
        "representative",
        "call me",
        "please call",
    ],

    "failed_support": [
        "already tried",
        "tried everything",
        "nothing works",
        "still not working",
        "still doesn't work",
        "still doesnt work",
        "didn't work",
        "didnt work",
        "tried that",
        "no luck",
    ],

    "account_action": [
        "close my account",
        "delete my account",
        "account locked",
        "can't login",
        "cant login",
        "can't log in",
        "cant log in",
        "reset my password",
        "change my account",
    ],

    "missing_delivery": [
        "marked delivered",
        "says delivered",
        "delivered but",
        "never arrived",
        "missing package",
        "missing order",
        "package missing",
        "not received",
        "didn't receive",
        "didnt receive",
    ],

    "high_frustration": [
        "ridiculous",
        "unacceptable",
        "worst",
        "terrible",
        "awful",
        "furious",
        "angry",
        "frustrated",
        "disappointed",
        "hate",
        "pathetic",
    ],
}


# ============================================================
# SCORING
# ============================================================

def count_matches(text, patterns):
    score = 0
    matched = []

    for pattern in patterns:
        if pattern in text:
            score += 1
            matched.append(pattern)

    return score, matched


def score_intents(current_text, context_text):
    """
    Current customer message gets more weight than historical context.
    Context is still used because many Amazon support messages are
    deictic / incomplete on their own.
    """

    current = clean_text(current_text)
    context = clean_text(context_text)

    scores = {}

    for intent, patterns in INTENT_PATTERNS.items():
        current_score, current_matches = count_matches(current, patterns)
        context_score, context_matches = count_matches(context, patterns)

        # Current message has 2x weight.
        total = (current_score * 2) + context_score

        scores[intent] = {
            "score": total,
            "current_matches": current_matches,
            "context_matches": context_matches,
        }

    return scores


def choose_candidate_intent(row):
    scores = score_intents(
        row["current_customer_message"],
        row["conversation_context"],
    )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True,
    )

    best_intent, best = ranked[0]
    second_score = ranked[1][1]["score"]

    # Do NOT force unmatched examples into information_general.
    if best["score"] == 0:
        return "unknown"

    # If two intents are very close, flag it as ambiguous.
    margin = best["score"] - second_score

    if best["score"] <= 2 and margin <= 1:
        return "ambiguous"

    return best_intent


def get_intent_metadata(row, intent):
    scores = score_intents(
        row["current_customer_message"],
        row["conversation_context"],
    )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True,
    )

    best_intent, best = ranked[0]
    second_score = ranked[1][1]["score"]

    return {
        "candidate_intent": intent,
        "candidate_score": best["score"],
        "candidate_second_score": second_score,
        "candidate_margin": best["score"] - second_score,
        "candidate_matches": "; ".join(
            best["current_matches"][:5]
            + best["context_matches"][:5]
        ),
    }


# ============================================================
# ACTION / ESCALATION / AMBIGUITY
# ============================================================

def candidate_action(row):
    text = (
        clean_text(row["current_customer_message"])
        + " "
        + clean_text(row["conversation_context"])
    )

    scores = {}

    for action, patterns in ACTION_PATTERNS.items():
        score, matches = count_matches(text, patterns)
        scores[action] = (score, matches)

    action, (score, matches) = max(
        scores.items(),
        key=lambda x: x[1][0]
    )

    if score == 0:
        action = "unknown"
        matches = []

    return action, matches


def escalation_flags(row):
    """
    Escalation sampling should primarily reflect what the
    CURRENT customer is asking for.

    Historical brand responses are not treated as evidence
    that the customer wants escalation.
    """

    current = clean_text(
        row["current_customer_message"]
    )

    context = clean_text(
        row["conversation_context"]
    )

    flags = []

    # --------------------------------------------------------
    # 1. Explicit human/support request
    # --------------------------------------------------------

    explicit_patterns = [
        "speak to a human",
        "speak to someone",
        "talk to someone",
        "real person",
        "human agent",
        "customer service agent",
        "representative",
        "call me",
        "please call",
        "i need a human",
        "i want a human",
        "let me speak",
    ]

    score, _ = count_matches(
        current,
        explicit_patterns
    )

    if score > 0:
        flags.append("explicit_human_request")

    # --------------------------------------------------------
    # 2. Failed troubleshooting/support
    # --------------------------------------------------------

    failed_patterns = [
        "already tried",
        "tried everything",
        "nothing works",
        "still not working",
        "still doesn't work",
        "still doesnt work",
        "didn't work",
        "didnt work",
        "tried that",
        "no luck",
    ]

    score, _ = count_matches(
        current,
        failed_patterns
    )

    if score > 0:
        flags.append("failed_support")

    # --------------------------------------------------------
    # 3. Account actions
    # --------------------------------------------------------

    account_patterns = [
        "close my account",
        "delete my account",
        "account locked",
        "can't login",
        "cant login",
        "can't log in",
        "cant log in",
        "reset my password",
        "change my account",
    ]

    score, _ = count_matches(
        current,
        account_patterns
    )

    if score > 0:
        flags.append("account_action")

    # --------------------------------------------------------
    # 4. Missing delivery
    #
    # Only flag strong missing-delivery language.
    # Generic "where is my order" belongs to delivery/status,
    # not automatic escalation.
    # --------------------------------------------------------

    missing_delivery_patterns = [
        "marked delivered",
        "says delivered",
        "delivered but",
        "never arrived",
        "missing package",
        "missing order",
        "package missing",
        "not received",
        "didn't receive",
        "didnt receive",
        "haven't received",
        "havent received",
    ]

    score, _ = count_matches(
        current,
        missing_delivery_patterns
    )

    if score > 0:
        flags.append("missing_delivery")

    # --------------------------------------------------------
    # 5. Strong frustration
    # --------------------------------------------------------

    frustration_patterns = [
        "ridiculous",
        "unacceptable",
        "worst service",
        "terrible service",
        "awful service",
        "furious",
        "extremely angry",
        "very angry",
        "extremely frustrated",
        "very frustrated",
        "pathetic service",
    ]

    score, _ = count_matches(
        current,
        frustration_patterns
    )

    if score > 0:
        flags.append("high_frustration")

    # --------------------------------------------------------
    # 6. Context-dependent unresolved case
    #
    # Context can strengthen an escalation candidate only
    # when the CURRENT message itself signals continuation
    # after a failed attempt.
    # --------------------------------------------------------

    unresolved_current = [
        "still",
        "again",
        "yet",
        "no luck",
        "didn't work",
        "didnt work",
        "still doesn't work",
        "still doesnt work",
    ]

    context_support_patterns = [
        "contact",
        "support",
        "help",
        "tried",
        "unable",
        "cannot",
        "can't",
        "issue",
        "problem",
    ]

    current_score, _ = count_matches(
        current,
        unresolved_current
    )

    context_score, _ = count_matches(
        context,
        context_support_patterns
    )

    if current_score > 0 and context_score > 0:
        flags.append("contextual_failed_support")

    return flags


def ambiguity_score(row):
    current = clean_text(row["current_customer_message"])
    context = clean_text(row["conversation_context"])

    score = 0
    reasons = []

    # Very short messages are often impossible to interpret reliably.
    if len(current.split()) <= 4:
        score += 2
        reasons.append("short_message")

    # Deictic phrases depend heavily on previous conversation.
    deictic = [
        "it",
        "this",
        "that",
        "they",
        "them",
        "still",
        "again",
        "there",
        "here",
    ]

    found_deictic = [x for x in deictic if re.search(rf"\b{x}\b", current)]

    if found_deictic:
        score += 2
        reasons.append("deictic_reference")

    # Context exists but current message is underspecified.
    if len(current.split()) <= 8 and len(context.split()) > 15:
        score += 2
        reasons.append("context_dependent")

    # Candidate classifier sees competing intents.
    scores = score_intents(
        row["current_customer_message"],
        row["conversation_context"],
    )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True,
    )

    if ranked[0][1]["score"] > 0:
        margin = ranked[0][1]["score"] - ranked[1][1]["score"]

        if margin <= 1:
            score += 2
            reasons.append("competing_intents")

    return score, reasons


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AmazonHelp Golden Set Sampler v2")
    print("=" * 70)

    df = pd.read_csv(INPUT)

    print(f"Loaded rows: {len(df):,}")

    required_columns = [
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_author_id",
        "current_customer_message",
        "historical_brand_response",
        "conversation_context",
        "context_turn_count",
        "customer_created_at",
        "brand_created_at",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

        # --------------------------------------------------------
    # Generate candidate intent
    # --------------------------------------------------------

    df["candidate_intent"] = df.apply(
        choose_candidate_intent,
        axis=1
    )

    metadata = df.apply(
        lambda row: get_intent_metadata(
            row,
            row["candidate_intent"]
        ),
        axis=1,
        result_type="expand",
    )

    # get_intent_metadata also returns candidate_intent,
    # but candidate_intent already exists in df.
    metadata = metadata[
        [
            "candidate_score",
            "candidate_second_score",
            "candidate_margin",
            "candidate_matches",
        ]
    ]

    df = pd.concat(
        [df, metadata],
        axis=1
    )

    # --------------------------------------------------------
    # Candidate action
    # --------------------------------------------------------
    actions = df.apply(
        candidate_action,
        axis=1,
        result_type="expand",
    )

    actions.columns = [
        "candidate_action",
        "candidate_action_matches",
    ]

    df = pd.concat([df, actions], axis=1)

    # --------------------------------------------------------
    # Escalation
    # --------------------------------------------------------

    df["escalation_flags"] = df.apply(
        lambda row: ";".join(
            escalation_flags(row)
        ),
        axis=1,
    )

    df["escalation_candidate"] = (
        df["escalation_flags"].str.len() > 0
    )

    # --------------------------------------------------------
    # Ambiguity
    # --------------------------------------------------------

    ambiguity = df.apply(
        ambiguity_score,
        axis=1,
        result_type="expand",
    )

    ambiguity.columns = [
        "ambiguity_score",
        "ambiguity_reasons",
    ]

    df = pd.concat([df, ambiguity], axis=1)

    df["ambiguity_candidate"] = (
        df["ambiguity_score"] >= 4
    )

    # --------------------------------------------------------
    # Remove exact duplicate customer messages
    #
    # This prevents the Golden Set from being dominated by
    # repeated copies of the same complaint.
    # --------------------------------------------------------

    df["_clean_current"] = df[
        "current_customer_message"
    ].map(clean_text)

    before = len(df)

    df = df[
        df["_clean_current"].str.len() > 0
    ].drop_duplicates(
        subset=["_clean_current"],
        keep="first"
    ).copy()

    print(
        f"Removed duplicate current messages: "
        f"{before - len(df):,}"
    )

    # --------------------------------------------------------
    # Create candidate buckets
    # --------------------------------------------------------

    candidate_columns = [
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_author_id",
        "current_customer_message",
        "historical_brand_response",
        "conversation_context",
        "context_turn_count",
        "customer_created_at",
        "brand_created_at",
        "candidate_intent",
        "candidate_score",
        "candidate_second_score",
        "candidate_margin",
        "candidate_matches",
        "candidate_action",
        "candidate_action_matches",
        "escalation_candidate",
        "escalation_flags",
        "ambiguity_candidate",
        "ambiguity_score",
        "ambiguity_reasons",
    ]

    # --------------------------------------------------------
    # Save individual candidate files
    # --------------------------------------------------------

    for intent in INTENTS:

        subset = df[
            df["candidate_intent"] == intent
        ].copy()

        # Harder examples first, then random diversity.
        subset = subset.sort_values(
            by=[
                "candidate_score",
                "ambiguity_score",
            ],
            ascending=[False, False],
        )

        subset.to_csv(
            OUTPUT_DIR / f"{intent}_candidates.csv",
            index=False,
        )

        print(
            f"{intent:30s} {len(subset):,}"
        )

    ambiguous = df[
        df["ambiguity_candidate"]
    ].copy()

    ambiguous = ambiguous.sort_values(
        by="ambiguity_score",
        ascending=False,
    )

    ambiguous.to_csv(
        OUTPUT_DIR / "ambiguous_candidates.csv",
        index=False,
    )

    escalation = df[
        df["escalation_candidate"]
    ].copy()

    escalation = escalation.sort_values(
        by="ambiguity_score",
        ascending=False,
    )

    escalation.to_csv(
        OUTPUT_DIR / "escalation_candidates.csv",
        index=False,
    )

    # Unknown examples are deliberately kept separate.
    unknown = df[
        df["candidate_intent"].isin(
            ["unknown", "ambiguous"]
        )
    ].copy()

    unknown.to_csv(
        OUTPUT_DIR / "unknown_or_ambiguous_candidates.csv",
        index=False,
    )

    # ========================================================
    # BUILD EXACT 200 EXAMPLES
    # ========================================================

    selected_ids = set()
    selected_parts = []

    # --------------------------------------------------------
    # STEP 1: balanced intent sample = 170
    #
    # For each intent:
    #   ~60% ordinary examples
    #   ~40% difficult examples
    # --------------------------------------------------------

    for intent, quota in QUOTAS.items():

        pool = df[
            (df["candidate_intent"] == intent)
            & (~df["customer_tweet_id"].isin(selected_ids))
            & (~df["ambiguity_candidate"])
            & (~df["escalation_candidate"])
        ].copy()

        if len(pool) == 0:
            print(
                f"WARNING: no candidates for {intent}"
            )
            continue

        hard_n = min(
            max(1, int(round(quota * 0.40))),
            len(pool)
        )

        ordinary_n = quota - hard_n

        # Hard pool: ambiguity, context, escalation.
        pool["hard_score"] = (
            pool["ambiguity_score"]
            + pool["escalation_candidate"].astype(int) * 2
            + (pool["context_turn_count"] >= 2).astype(int)
        )

        hard_pool = pool.sort_values(
            "hard_score",
            ascending=False
        ).head(hard_n)

        hard_ids = set(
            hard_pool["customer_tweet_id"]
        )

        remaining = pool[
            ~pool["customer_tweet_id"].isin(hard_ids)
        ]

        if len(remaining) > 0:
            ordinary = remaining.sample(
                n=min(ordinary_n, len(remaining)),
                random_state=SEED,
            )
        else:
            ordinary = pd.DataFrame()

        part = pd.concat(
            [hard_pool, ordinary],
            ignore_index=True,
        )

        # Backfill if bucket had too few examples.
        if len(part) < quota:

            remaining = pool[
                ~pool["customer_tweet_id"].isin(
                    set(part["customer_tweet_id"])
                )
            ]

            extra = remaining.head(
                quota - len(part)
            )

            part = pd.concat(
                [part, extra],
                ignore_index=True,
            )

        part = part.head(quota)

        selected_parts.append(part)

        selected_ids.update(
            part["customer_tweet_id"]
        )

    # --------------------------------------------------------
    # STEP 2: explicit ambiguous examples = 15
    # --------------------------------------------------------

    ambiguous_pool = df[
        df["ambiguity_candidate"]
        & (~df["customer_tweet_id"].isin(selected_ids))
    ].copy()
    ambiguous_pool = ambiguous_pool.sort_values(
    by=[
        "ambiguity_score",
        "context_turn_count",
    ],
    ascending=[False, False],)
    ambiguous_sample = ambiguous_pool.head(
    AMBIGUOUS_QUOTA)
    

    selected_parts.append(
        ambiguous_sample
    )

    selected_ids.update(
        ambiguous_sample["customer_tweet_id"]
    )

    # --------------------------------------------------------
    # STEP 3: escalation-heavy examples = 15
    # --------------------------------------------------------

    escalation_pool = df[
        df["escalation_candidate"]
        & (~df["ambiguity_candidate"])
        & (~df["customer_tweet_id"].isin(selected_ids))
    ].copy()

    escalation_pool["escalation_strength"] = (
        escalation_pool["escalation_flags"]
        .str.count(";")
        + 1
    )

    escalation_pool = escalation_pool.sort_values(
        by=[
            "escalation_strength",
            "ambiguity_score",
        ],
        ascending=False,
    )

    escalation_sample = escalation_pool.head(
        ESCALATION_QUOTA
    )

    selected_parts.append(
        escalation_sample
    )

    selected_ids.update(
        escalation_sample["customer_tweet_id"]
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    golden = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Exact-size backfill
    # --------------------------------------------------------

    if len(golden) < RECOMMENDED_SIZE:

        remaining = df[
            ~df["customer_tweet_id"].isin(selected_ids)
        ].copy()

        # Prefer context-rich remaining examples.
        remaining["backfill_score"] = (
            remaining["context_turn_count"]
            + remaining["ambiguity_score"]
            + remaining["escalation_candidate"].astype(int)
        )

        remaining = remaining.sort_values(
            "backfill_score",
            ascending=False,
        )

        extra_n = RECOMMENDED_SIZE - len(golden)

        extra = remaining.head(extra_n)

        golden = pd.concat(
            [golden, extra],
            ignore_index=True,
        )

        selected_ids.update(
            extra["customer_tweet_id"]
        )

    # If somehow >200 due to unusual missing buckets,
    # keep the first 200 deterministically.
    golden = golden.head(
        RECOMMENDED_SIZE
    ).copy()

    # --------------------------------------------------------
    # Assign sampling reason
    # --------------------------------------------------------

    def sampling_reason(row):

        reasons = []

        if row["ambiguity_candidate"]:
            reasons.append("ambiguous")

        if row["escalation_candidate"]:
            reasons.append("escalation")

        if row["context_turn_count"] >= 2:
            reasons.append("context_heavy")

        if not reasons:
            reasons.append("balanced_intent")

        return ";".join(reasons)

    golden["sampling_reason"] = golden.apply(
        sampling_reason,
        axis=1,
    )

    # --------------------------------------------------------
    # Add stable evaluation ID
    # --------------------------------------------------------

    golden.insert(
        0,
        "golden_id",
        [
            f"AMZ-{i:04d}"
            for i in range(1, len(golden) + 1)
        ],
    )

    # --------------------------------------------------------
    # Save recommended Golden Set
    # --------------------------------------------------------

    golden_path = OUTPUT_DIR / "golden_set_sampling_v2.csv"

    golden.to_csv(
        golden_path,
        index=False,
    )

    # ========================================================
    # BUILD BACKUP POOL
    # ========================================================

    backup_pool = df[
        ~df["customer_tweet_id"].isin(
            set(golden["customer_tweet_id"])
        )
    ].copy()

    backup_pool["backup_priority"] = (
        backup_pool["ambiguity_score"]
        + backup_pool["escalation_candidate"].astype(int) * 2
        + (backup_pool["context_turn_count"] >= 2).astype(int)
    )

    # Take difficult cases first, then random diversity.
    hard_backup = backup_pool.sort_values(
        "backup_priority",
        ascending=False,
    ).head(
        min(300, len(backup_pool))
    )

    hard_ids = set(
        hard_backup["customer_tweet_id"]
    )

    random_backup_pool = backup_pool[
        ~backup_pool["customer_tweet_id"].isin(hard_ids)
    ]

    random_n = min(
        BACKUP_SIZE - len(hard_backup),
        len(random_backup_pool),
    )

    if random_n > 0:
        random_backup = random_backup_pool.sample(
            n=random_n,
            random_state=SEED,
        )
    else:
        random_backup = pd.DataFrame()

    backup = pd.concat(
        [hard_backup, random_backup],
        ignore_index=True,
    ).head(BACKUP_SIZE)

    backup.insert(
        0,
        "backup_id",
        [
            f"BK-{i:04d}"
            for i in range(1, len(backup) + 1)
        ],
    )

    backup.to_csv(
        OUTPUT_DIR / "golden_set_backup_pool_v2.csv",
        index=False,
    )

    # ========================================================
    # LABELING TEMPLATE
    # ========================================================

    labeling_columns = [
        "golden_id",
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_author_id",
        "current_customer_message",
        "historical_brand_response",
        "conversation_context",
        "context_turn_count",

        # Human labels — intentionally blank
        "gold_intent",
        "gold_action",
        "gold_handling",
        "gold_reason",
        "gold_evidence_sufficient",

        # Sampling metadata
        "sampling_reason",
        "candidate_intent",
        "candidate_action",
        "ambiguity_score",
        "escalation_flags",
    ]

    labeling = golden.copy()
    labeling["gold_intent"] = ""
    labeling["gold_action"] = ""
    labeling["gold_handling"] = ""
    labeling["gold_reason"] = ""
    labeling["gold_evidence_sufficient"] = ""

    labeling = labeling[
    labeling_columns].copy()

    labeling_path = (
        OUTPUT_DIR
        / "golden_set_labeling_template_v2.csv"
    )

    labeling.to_csv(
        labeling_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("V2 SAMPLING COMPLETE")
    print("=" * 70)

    print(
        f"\nRecommended Golden Set: "
        f"{len(golden):,}"
    )

    print(
        f"Backup pool: "
        f"{len(backup):,}"
    )

    print("\nRecommended candidate-intent distribution:")

    print(
        golden["candidate_intent"]
        .value_counts()
        .to_string()
    )

    print("\nSampling reason distribution:")

    print(
        golden["sampling_reason"]
        .value_counts()
        .to_string()
    )

    print("\nEscalation candidates in Golden Set:")

    print(
        golden["escalation_candidate"]
        .value_counts()
        .to_string()
    )

    print("\nAmbiguous candidates in Golden Set:")

    print(
        golden["ambiguity_candidate"]
        .value_counts()
        .to_string()
    )

    print("\nContext distribution:")

    print(
        golden["context_turn_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFiles created:")

    print(f"  {golden_path}")
    print(
        f"  {OUTPUT_DIR / 'golden_set_labeling_template_v2.csv'}"
    )
    print(
        f"  {OUTPUT_DIR / 'golden_set_backup_pool_v2.csv'}"
    )

    print("\nIMPORTANT:")
    print(
        "candidate_intent/candidate_action are sampling hints only."
    )
    print(
        "Do NOT use them as ground truth."
    )
    print(
        "Only gold_* fields should be used as evaluation labels."
    )


if __name__ == "__main__":
    main()