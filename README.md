# Hiver SDE Intern — Evidence-Grounded Support Agent

A runnable, local-first reference implementation for the Hiver take-home assignment. It demonstrates the core idea: **build an AI support agent whose failure modes are measurable and explicit**.

## What it does

For an incoming customer-support message, the pipeline:

1. Classifies the message into a small brand-specific intent taxonomy.
2. Retrieves similar historical resolved cases.
3. Drafts a response grounded in those cases.
4. Decides `AUTO_HANDLE` vs `ESCALATE` and gives a reason.
5. Evaluates intent quality, retrieval quality, response grounding, and escalation behavior.

The default demo is fully offline and uses TF-IDF + Logistic Regression + cosine retrieval, so it runs without an API key. An LLM can be plugged into the response generator later.

## Quick start (<15 min)

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.cli --message "My order still hasn't arrived and I want an update"
python -m src.cli --evaluate
```

## Expected demo output

The exact metrics depend on the sample data. The system prints an intent, confidence, retrieved historical evidence, drafted response, escalation decision, and evaluation metrics.

## Using the real Kaggle dataset

Download the Customer Support on Twitter dataset separately and place its CSV under `data/raw/`. The loader accepts common column names such as `text`, `tweet_id`, `in_response_to_tweet_id`, and `author_id`. Run:

```bash
python -m src.cli --prepare-real-data data/raw/your_file.csv --brand "@brand"
```

Because the original corpus is large and noisy, this project intentionally works on a filtered/sample subset. The submission should report exactly how the subset was created.

## Architecture

```text
Customer message
      |
      v
Intent classifier -----> confidence
      |
      v
Historical-case retriever
      |
      v
Grounded response generator
      |
      v
Escalation policy
      |
  +---+---+
  |       |
AUTO    HUMAN

Evaluation harness surrounds the pipeline and measures each component.
```

## Default intent taxonomy

The sample brand uses 8 intentionally small intents:

- delivery_issue
- refund_request
- payment_issue
- order_issue
- cancellation
- account_issue
- promotion_query
- other

For the real submission, replace these with a taxonomy discovered from the selected brand's actual conversations.

## Failure modes intentionally handled

The agent is conservative when:

- confidence is low;
- no sufficiently similar historical resolution exists;
- the message contains a safety/legal/financial-risk trigger;
- the customer indicates a repeated unresolved problem;
- the request is too ambiguous to answer safely.

These are not treated as model "failures" by default. Escalation is a valid system outcome when the evidence is insufficient.

## Important evaluation note

Do not use the demo metrics in a submission. They exist only to prove the pipeline works. Replace `evaluation/golden_set.jsonl` with your 150–250 manually labelled examples and run the harness on that set.

## Suggested next steps for the actual assignment

1. Select one brand based on resolution density and conversation quality.
2. Reconstruct multi-turn threads.
3. Discover 8–12 intents from that brand.
4. Create a 150–250 example Golden Set with a separate hard-case slice.
5. Add TF-IDF/LR and majority-class baselines.
6. Replace/augment the local generator with an LLM and require historical evidence.
7. Add an LLM judge with a 0–4 rubric for correctness, groundedness, helpfulness, brand consistency, and safety.
8. Manually score ~50 responses and report judge/human agreement.
9. Document the top five failure modes with real examples.
