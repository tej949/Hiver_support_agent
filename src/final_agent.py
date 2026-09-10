"""
Final local support agent for the Hiver take-home.

Pipeline:
  1. Thread-aware context
  2. Intent classification (TF-IDF + Logistic Regression)
  3. Historical case retrieval, restricted to training data
  4. Evidence sufficiency / similarity gate
  5. Grounded response drafting from historical resolution patterns
  6. AUTO-HANDLE vs ESCALATE decision with an explicit reason

This version is API-free and reproducible. It deliberately does not copy a
historical customer's reply verbatim; it extracts reusable resolution guidance
and builds a short template response.
"""

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path("data/processed")
THREAD = BASE / "amazonhelp_thread_context.csv"
TRAIN = BASE / "splits/train_silver.csv"
GOLDEN = BASE / "golden_v2/golden_set_locked.csv"
OUT = BASE / "final_agent"
OUT.mkdir(parents=True, exist_ok=True)

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

ACTIONS = [
    "self_service",
    "investigation",
    "phone_chat",
    "callback",
    "refund_replacement",
    "information",
    "acknowledgement",
    "privacy_restriction",
    "other",
]


def norm_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def normalize_text(text):
    text = str(text or "")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_context(row):
    parts = []
    for n in (1, 2, 3):
        c = str(row.get(f"prev_{n}_customer", "") or "").strip()
        b = str(row.get(f"prev_{n}_brand", "") or "").strip()
        if c:
            parts.append(f"Customer: {c}")
        if b:
            parts.append(f"AmazonHelp: {b}")
    return "\n".join(parts)


def combined_text(row):
    current = normalize_text(row["customer_text"])
    context = normalize_text(make_context(row))
    # Current message is intentionally weighted more strongly than context.
    return (current + " ") * 2 + context


def escalation_reason(text, predicted_intent, evidence_score):
    t = text.lower()

    strong = [
        "hacked", "compromised", "unauthorized", "can't login", "cannot login",
        "locked out", "close my account", "stolen", "fraud",
    ]
    unresolved = [
        "still", "again", "already", "multiple", "repeated", "days",
        "week", "weeks", "month", "months", "not received", "not arrived",
        "not delivered", "missing", "no refund", "refund not", "failed",
        "nobody", "no one", "complaint", "call me", "phone number",
    ]
    sensitive = [
        "card", "bank", "payment", "charge", "cashback", "money",
    ]

    if any(x in t for x in strong):
        return True, "Account/security issue requires private account-level handling."
    if predicted_intent == "account_access" and any(x in t for x in unresolved):
        return True, "Account access remains unresolved and needs account-level intervention."
    if predicted_intent in {"delivery", "order_status_tracking"} and any(x in t for x in unresolved):
        return True, "The order appears unresolved or overdue and needs order-specific investigation."
    if predicted_intent in {"payment_billing", "return_refund_replacement"} and any(x in t for x in unresolved):
        return True, "A payment/refund issue remains unresolved and needs case-specific investigation."
    if predicted_intent == "customer_service_complaint":
        return True, "The customer is reporting unresolved support and should be routed to a human."
    if evidence_score < 0.25:
        return True, "No sufficiently similar historical resolution was retrieved, so a grounded reply is unsafe."
    return False, "The request appears answerable from relevant historical support guidance."


def choose_action(response_text):
    r = response_text.lower()

    if any(x in r for x in ["call us", "phone", "contact us", "chat with us", "live chat"]):
        return "phone_chat"
    if any(x in r for x in ["refund", "replacement", "replace", "reimburse"]):
        return "refund_replacement"
    if any(x in r for x in ["fill out", "form", "help page", "visit us here", "troubleshoot", "steps"]):
        return "self_service"
    if any(x in r for x in ["escalate", "investigate", "look into", "check this for you"]):
        return "investigation"
    if any(x in r for x in ["sorry", "apolog", "thanks", "glad", "welcome"]):
        return "acknowledgement"
    return "information"


def extract_urls(text):
    return re.findall(r"https?://\S+", str(text or ""))


def draft_reply(current, intent, evidence, escalate):
    ev_text = evidence.get("brand_response", "")
    action = choose_action(ev_text)
    urls = extract_urls(ev_text)

    if intent in {"delivery", "order_status_tracking"}:
        base = "I’m sorry your order is still unresolved. Please check the latest order/tracking status and, if the promised delivery date has passed, contact support so the order can be investigated."
    elif intent == "return_refund_replacement":
        base = "I’m sorry this return/refund or replacement is still unresolved. Please use the return/refund options for the order, and contact support if the expected resolution has not completed."
    elif intent == "payment_billing":
        base = "I’m sorry about the billing issue. Please review the transaction/order details and contact support if the charge, refund, or cashback is still unresolved so it can be checked against the account."
    elif intent == "account_access":
        base = "For account-access issues, please use the account recovery/sign-in support flow. If you still cannot access the account, support will need to verify the account privately."
    elif intent == "product_device_issue":
        base = "Please try the relevant product troubleshooting steps first. If the issue persists, product support can investigate the device or order-specific case."
    elif intent == "shipping_delivery_options":
        base = "Please check the available delivery options and the estimated delivery date for the address/order. If a paid or promised shipping service was not provided, support can review the order."
    elif intent == "order_change_cancel":
        base = "Please check the order’s current status for the available change or cancellation options. If it has already shipped, support may need to review what options remain."
    elif intent == "customer_service_complaint":
        base = "I’m sorry this has not been resolved. This needs a support representative to review the case and the previous contact history."
    else:
        base = "Here’s the relevant guidance based on similar AmazonHelp cases: please use the applicable Amazon help/support flow for this request."

    if urls:
        base += f" Relevant help: {urls[0]}"

    if escalate:
        base += " Because this is unresolved or account/order-specific, I’d route it to a human rather than make an unsupported promise."

    return base, action


class FinalAgent:
    def __init__(self):
        self.thread = pd.read_csv(THREAD, dtype=str, low_memory=False).fillna("")
        self.train = pd.read_csv(TRAIN, dtype=str, low_memory=False).fillna("")

        # Train intent model only on silver training data.
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        X = self.vectorizer.fit_transform(self.train["customer_text"].map(normalize_text))
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.classifier.fit(X, self.train["gold_intent"])

        # Retrieval corpus is restricted to the training customers/rows.
        train_ids = set(self.train["customer_tweet_id"].map(norm_id))
        self.retrieval = self.thread[
            self.thread["customer_tweet_id"].map(norm_id).isin(train_ids)
        ].copy()

        self.retrieval["retrieval_text"] = (
            self.retrieval["customer_text"].map(normalize_text)
            + " "
            + self.retrieval.apply(make_context, axis=1).map(normalize_text)
        )
        self.retrieval["intent"] = self.retrieval["customer_tweet_id"].map(
            self.train.set_index("customer_tweet_id")["gold_intent"].to_dict()
        )

        self.retrieval_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.R = self.retrieval_vectorizer.fit_transform(self.retrieval["retrieval_text"])

    def predict(self, message, context=""):
        clf_text = normalize_text(message)
        X = self.vectorizer.transform([clf_text])
        probs = self.classifier.predict_proba(X)[0]
        order = np.argsort(probs)[::-1]
        intent = self.classifier.classes_[order[0]]
        confidence = float(probs[order[0]])

        retrieval_query = normalize_text(message) + " " + normalize_text(context)
        q = self.retrieval_vectorizer.transform([retrieval_query])

        # First retrieve broadly, then prefer same-intent historical cases.
        scores = cosine_similarity(q, self.R).ravel()
        candidates = np.argsort(scores)[::-1][:30]

        same_intent = [i for i in candidates if self.retrieval.iloc[i]["intent"] == intent]
        chosen_idx = same_intent[0] if same_intent else candidates[0]
        evidence_score = float(scores[chosen_idx])
        ev = self.retrieval.iloc[chosen_idx].to_dict()

        escalate, reason = escalation_reason(message, intent, evidence_score)
        reply, action = draft_reply(message, intent, ev, escalate)

        return {
            "intent": intent,
            "intent_confidence": round(confidence, 4),
            "retrieval_similarity": round(evidence_score, 4),
            "evidence_sufficient": "YES" if evidence_score >= 0.25 else "NO",
            "action": action,
            "handling": "ESCALATE" if escalate else "AUTO-HANDLE",
            "reason": reason,
            "reply": reply,
            "evidence_customer": ev.get("customer_text", ""),
            "evidence_response": ev.get("brand_response", ""),
            "evidence_customer_tweet_id": ev.get("customer_tweet_id", ""),
            "evidence_brand_tweet_id": ev.get("brand_tweet_id", ""),
        }


def evaluate_golden(agent):
    if not GOLDEN.exists():
        print(f"Golden set not found: {GOLDEN}")
        return

    gold = pd.read_csv(GOLDEN, dtype=str, low_memory=False).fillna("")
    rows = []

    for _, r in gold.iterrows():
        context = make_context(r) if "prev_1_customer" in gold.columns else ""
        pred = agent.predict(r["customer_text"], context)
        rows.append({
            "customer_tweet_id": r.get("customer_tweet_id", ""),
            "gold_intent": r["gold_intent"],
            "pred_intent": pred["intent"],
            "gold_action": r["gold_action"],
            "pred_action": pred["action"],
            "gold_handling": r["gold_handling"],
            "pred_handling": pred["handling"],
            "gold_evidence_sufficient": r["gold_evidence_sufficient"],
            "pred_evidence_sufficient": pred["evidence_sufficient"],
            "intent_confidence": pred["intent_confidence"],
            "retrieval_similarity": pred["retrieval_similarity"],
            "reason": pred["reason"],
            "reply": pred["reply"],
            "evidence_customer": pred["evidence_customer"],
            "evidence_response": pred["evidence_response"],
            "evidence_customer_tweet_id": pred["evidence_customer_tweet_id"],
            "evidence_brand_tweet_id": pred["evidence_brand_tweet_id"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "golden_predictions.csv", index=False)

    from sklearn.metrics import accuracy_score, f1_score, classification_report

    metrics = {
        "n": len(out),
        "intent_accuracy": float(accuracy_score(out.gold_intent, out.pred_intent)),
        "intent_macro_f1": float(f1_score(out.gold_intent, out.pred_intent, average="macro")),
        "action_accuracy": float(accuracy_score(out.gold_action, out.pred_action)),
        "action_macro_f1": float(f1_score(out.gold_action, out.pred_action, average="macro")),
        "handling_accuracy": float(accuracy_score(out.gold_handling, out.pred_handling)),
        "handling_macro_f1": float(f1_score(out.gold_handling, out.pred_handling, average="macro")),
        "evidence_agreement": float(
            accuracy_score(out.gold_evidence_sufficient, out.pred_evidence_sufficient)
        ),
        "escalation_precision": float(
            __import__("sklearn.metrics").metrics.precision_score(
                out.gold_handling, out.pred_handling,
                pos_label="ESCALATE", zero_division=0
            )
        ),
        "escalation_recall": float(
            __import__("sklearn.metrics").metrics.recall_score(
                out.gold_handling, out.pred_handling,
                pos_label="ESCALATE", zero_division=0
            )
        ),
    }

    (OUT / "golden_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=" * 70)
    print("FINAL AGENT — GOLDEN SET EVALUATION")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nIntent report:")
    print(classification_report(out.gold_intent, out.pred_intent, zero_division=0))

    print("Handling report:")
    print(classification_report(out.gold_handling, out.pred_handling, zero_division=0))

    print(f"Saved predictions: {OUT / 'golden_predictions.csv'}")
    print(f"Saved metrics:     {OUT / 'golden_metrics.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, help="Run one message through the agent")
    parser.add_argument("--context", type=str, default="", help="Optional thread context")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on 200-row Golden Set")
    args = parser.parse_args()

    agent = FinalAgent()

    if args.evaluate:
        evaluate_golden(agent)

    if args.message:
        result = agent.predict(args.message, args.context)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
