import argparse, json
from pathlib import Path
from .agent import SupportAgent
from .evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data/sample/resolved_cases.jsonl"
CONFIG = ROOT / "configs/config.yaml"
GOLDEN = ROOT / "evaluation/golden_set.jsonl"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--message")
    p.add_argument("--evaluate", action="store_true")
    args = p.parse_args()
    agent = SupportAgent(CASES, CONFIG)
    if args.message:
        print(json.dumps(agent.run(args.message), indent=2))
    if args.evaluate:
        result = evaluate(agent, GOLDEN)
        print("\n=== EVALUATION ===")
        print(f"N: {result['n']}")
        print(f"Accuracy: {result['accuracy']:.3f}")
        print(f"Macro F1: {result['macro_f1']:.3f}")
        print(f"Weighted F1: {result['weighted_f1']:.3f}")
        print(result["report"])
    if not args.message and not args.evaluate:
        p.print_help()

if __name__ == "__main__":
    main()
