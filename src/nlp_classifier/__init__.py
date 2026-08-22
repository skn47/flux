"""
Phase C: event-type classifier (trained from scratch), parallel to
`ingestion/` (Phase A) and `labeling/` (Phase B).

Scope call (documented in nlp_classifier/README.md): this package trains and
evaluates an **event-type classifier only**. Country/sector/company
extraction remains the deterministic keyword matching already implemented in
`labeling/rule_labeler.py` -- there is currently zero span-level (entity
boundary) annotated data in this project, so training a real extractive NER
model has nothing to learn from. That is flagged as explicit future work,
not silently dropped.
"""
