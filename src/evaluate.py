import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
from .agent import SupportAgent

def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def evaluate(agent, golden_path):
    rows = load_jsonl(golden_path)
    y_true, y_pred = [], []
    details = []
    for row in rows:
        out = agent.run(row["text"])
        y_true.append(row["intent"])
        y_pred.append(out["intent"])
        details.append({"gold": row["intent"], "pred": out["intent"], "text": row["text"], "action": out["action"]})
    return {
        "n": len(rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "report": classification_report(y_true, y_pred, zero_division=0),
        "details": details,
    }
