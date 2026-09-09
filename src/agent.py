from dataclasses import dataclass
from pathlib import Path
import json
import re
import yaml
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Prediction:
    intent: str
    confidence: float

class SupportAgent:
    def __init__(self, cases_path, config_path):
        self.cases = [json.loads(x) for x in Path(cases_path).read_text(encoding="utf-8").splitlines() if x.strip()]
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.classifier = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True)),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        texts = [c["customer"] for c in self.cases]
        labels = [c["intent"] for c in self.cases]
        self.classifier.fit(texts, labels)
        self.retriever = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
        self.case_matrix = self.retriever.fit_transform(texts)

    def classify(self, text):
        probs = self.classifier.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        return Prediction(str(self.classifier.classes_[idx]), float(probs[idx]))

    def retrieve(self, text, top_k=None):
        top_k = top_k or self.config["retrieval"]["top_k"]
        q = self.retriever.transform([text])
        sims = cosine_similarity(q, self.case_matrix)[0]
        order = np.argsort(-sims)[:top_k]
        return [(self.cases[i], float(sims[i])) for i in order]

    def escalation(self, text, prediction, evidence):
        low_conf = prediction.confidence < self.config["classifier"]["min_confidence"]
        weak_evidence = not evidence or evidence[0][1] < self.config["retrieval"]["min_similarity"]
        lowered = text.lower()
        risk_hit = next((t for t in self.config["escalation"]["risk_terms"] if t in lowered), None)
        unresolved_hit = next((t for t in self.config["escalation"]["unresolved_followup_terms"] if t in lowered), None)
        if risk_hit:
            return "ESCALATE", f"High-risk signal detected: '{risk_hit}'."
        if low_conf:
            return "ESCALATE", f"Low intent confidence ({prediction.confidence:.2f})."
        if weak_evidence:
            return "ESCALATE", "No sufficiently similar historical resolution was found."
        if unresolved_hit:
            return "ESCALATE", f"Possible repeated/unresolved contact: '{unresolved_hit}'."
        return "AUTO_HANDLE", "Intent is confident and a similar historical resolution is available."

    def draft_reply(self, text, prediction, evidence):
        if not evidence or evidence[0][1] < self.config["retrieval"]["min_similarity"]:
            return "Thanks for reaching out. I want to make sure we give you the right answer. Please share a few more details so a support specialist can help."
        best, score = evidence[0]
        reply = best["brand_reply"]
        # Prevent the demo from blindly copying a historical response for a different intent.
        if best["intent"] != prediction.intent:
            return "Thanks for reaching out. Please share the order or account details relevant to your issue so we can check this accurately."
        return reply

    def run(self, text):
        pred = self.classify(text)
        evidence = self.retrieve(text)
        action, reason = self.escalation(text, pred, evidence)
        reply = self.draft_reply(text, pred, evidence)
        return {
            "message": text,
            "intent": pred.intent,
            "confidence": round(pred.confidence, 4),
            "action": action,
            "reason": reason,
            "draft_reply": reply,
            "evidence": [
                {"intent": c["intent"], "similarity": round(s, 4), "customer": c["customer"], "brand_reply": c["brand_reply"]}
                for c, s in evidence
            ],
        }
