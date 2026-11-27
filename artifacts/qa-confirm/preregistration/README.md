# QA confirmatory scheduler study: frozen protocol

Committed before any confirmatory test article was scored.

- `protocol.json`: hypothesis, frozen policy, cohort rules, arms, outcomes, inference, decision gate, sample size and amendment A1.
- `cohort.json`: seeded draw (seed 20261127) with every candidate article's outcome and exclusion reason. 268 accepted articles: 20 for development and 248 for the test split (113 numeric, 135 capitalized).
- `screen-<model>-<style>.json`: development-split clean QA and direct text-edit screens for SmolLM2-1.7B and SmolLM2-360M under prompt styles v0 to v3. No intervention arm was run. Selected: SmolLM2-1.7B, style v2, with 4 of 20 usable articles.
- `validate-*.json`: smoke and addressing checks on development articles. They cover self-donor identity, exact self logits, bitwise capture, repeat and no-op identity, linearity, mask-versus-write, output-projection reconstruction and wrong-head contrast. They also check forward accounting and, for Qwen, equivalence with the recorded pilot online rows.

Under bf16, SmolLM2's linearity and projection residuals are large only when the deltas are small; the float32 run of the same code gives residuals of 6e-4 or less. Qwen's layer-3 linearity residual reflects the activation quantization of llama.cpp.

Code: `scripts/query_transition/qa_confirm*.py` and `qa_confirm_probe.cpp`.
