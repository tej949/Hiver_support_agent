# Hiver Take-Home Report Template

## 1. Problem framing

**Brand:** [BRAND]

**What good means:** [Define useful, accurate, grounded, and safe support for this brand.]

**Not built:** autonomous refunds, account changes, external actions, or direct message sending.

## 2. Data and Golden Set

- Source: Customer Support on Twitter.
- Brand filtering method: [describe].
- Thread reconstruction: [describe].
- Sample size: [N].
- Golden Set: [150–250 examples].
- Hard-set composition: [describe].

## 3. Results

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Majority class | [ ] | [ ] |
| TF-IDF + Logistic Regression | [ ] | [ ] |
| LLM / final system | [ ] | [ ] |

Also report:
- escalation precision / recall;
- reply groundedness;
- helpfulness;
- unsafe auto-handling rate.

## 4. LLM-as-judge validation

Rubric dimensions: correctness, groundedness, helpfulness, brand consistency, safety.

Human sample: [N].

Judge-human agreement: [metric and value].

## 5. Failure analysis

### Failure 1 — [name]
Example: [real anonymized example]

Why it failed: [ ]

Hypothesis: [ ]

Repeat for top 5.

## 6. What is misleading about my headline number?

[Explain why the headline metric does not mean the same percentage of arbitrary customer conversations are safe to auto-handle. Discuss class imbalance, hard cases, OOD inputs, and escalation.]

## 7. One more week

1. [ ]
2. [ ]
3. [ ]
