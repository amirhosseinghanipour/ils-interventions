# Query transition measurements

These arrays and JSON records accompany the ILS Interventions experiments. [Reproduction guide](https://github.com/amirhosseinghanipour/ils-interventions/tree/main/scripts/query_transition) Each directory contains results for the 9B and 4B GGUF checkpoints. Block indices start at zero. Two directions of intervention from a single prefix share a resampling unit.

* `blocks` contains paired two-word questions, clean logits and finite interventions from blocks 16 through 23. Compressed or uncompressed NumPy archives contain the two final positions of each clean forward pass and the food-task scans. File sizes differ by model width.
* `heads` contains interventions on the two-word development and four-person assembly tasks, head selection scores, and the post-gate activations for the selected block.
* `transport` contains block-edit displacements paired with their final residual responses and out-of-sample prediction scores.
* `SHA256SUMS` lists the content digests of all measured files.
