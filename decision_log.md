# Decision log

1. **One small intent taxonomy:** A compact taxonomy is easier to evaluate and more useful for routing than dozens of overlapping labels.
2. **Local-first baseline:** The reference implementation runs without an API key so reviewers can reproduce it quickly.
3. **TF-IDF + Logistic Regression:** Chosen as an interpretable, strong classical baseline.
4. **Separate retrieval representation:** Retrieval and classification have different objectives, so they use separate TF-IDF matrices.
5. **Historical resolution evidence:** Responses are generated from resolved examples instead of unconstrained free-form generation.
6. **Conservative evidence threshold:** Weak retrieval evidence triggers escalation instead of invented policy.
7. **Explicit escalation reasons:** Human routing must be explainable to an operator.
8. **Risk triggers:** Legal, fraud, hacking, and safety language is escalated regardless of classifier confidence.
9. **Repeated-contact trigger:** Phrases indicating prior unresolved contact increase escalation likelihood.
10. **No autonomous account/refund actions:** The system drafts and routes; it does not execute sensitive operations.
11. **Macro-F1:** Reported alongside accuracy because intent frequencies can be highly imbalanced.
12. **Golden set separated from training cases:** Evaluation examples must not simply be copied from the historical retrieval corpus.
13. **Hard cases should be oversampled for analysis:** Ambiguous and multi-turn follow-ups expose trust failures that average accuracy can hide.
14. **Headline metric caveat:** Accuracy does not equal safe automation rate; escalation quality and groundedness matter separately.
