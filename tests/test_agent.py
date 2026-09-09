from pathlib import Path
from src.agent import SupportAgent

ROOT = Path(__file__).resolve().parents[1]

def test_delivery():
    a=SupportAgent(ROOT/'data/sample/resolved_cases.jsonl', ROOT/'configs/config.yaml')
    o=a.run('Where is my package?')
    assert o['intent']=='delivery_issue'

def test_risky_message_escalates():
    a=SupportAgent(ROOT/'data/sample/resolved_cases.jsonl', ROOT/'configs/config.yaml')
    o=a.run("Nobody helped me and I'm going to sue")
    assert o['action']=='ESCALATE'
