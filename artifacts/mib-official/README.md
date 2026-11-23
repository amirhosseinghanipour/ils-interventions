# Public GPT-2 IOI circuit scores

The 1,000 public IOI test prompts have nine templates disjoint from the development split. Every test prompt receives the same graph at a given edge budget and objective. CPR rewards a larger recovered answer margin. CMD rewards closeness to the full model. Each score integrates ten sparsities. The prompt-local head-group experiment has a separate protocol and outcome. [Reproduce these scores](../../scripts/mib_official/README.md) · [Audit the prompt-level measurements](ioi-public-test.json.gz)

| Method | CPR (higher) | CMD (lower) |
| --- | ---: | ---: |
| Organizer EAP-IG example | 0.2414 | 0.0264 |
| Released MAttr | 7.9428 | 0.1212 |
| Released Technion ILP | 1.8828 | 0.0152 |
| Best measured edge-band exchange | 0.2448 | 0.0269 |
| Additive edge-band fit | 0.2433 | 0.0256 |
| Pairwise edge-band fit | 0.2435 | 0.0261 |
| Organizer submission example (post-primary) | 1.2013 | 0.0342 |

For each nontrivial sparsity and objective, 16 development prompts determine the selected edge-band exchange. The fits start from the organizer's released example ranking. Its training-set provenance is unspecified. The MAttr and Technion results use their released artifacts. The graph fit uses separate circuits for CPR and CMD. [Per-sparsity results and paired template intervals](public-summary.json) · [Edge and node mask audit](mask-audit.json)

A separate organizer HF submission example was added after the primary policy freeze. Its signed edge scores differ from the GitHub example used for development. [Additional native scores and paired intervals](organizer-hf-public-summary.json) · [Additional prompt-level margins](organizer-hf-public-test.json.gz)

The organizer's dashboard and older archived EAP-IG curve describe separate evaluations. Neither is used to score this comparison. [Dashboard and archive record](published-scorecard.json) · [File checksums](SHA256SUMS)
