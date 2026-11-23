# Query-routing experiment data

The v1 directory contains the paired synthetic prompts, tokens, observed logits and fp32 residual changes from the Qwen3.5-9B Q5_K_M experiment. Each NPZ stores both clean final residual trajectories and signed intervention changes relative to the corresponding clean trajectory. The last row is the first-answer prediction site. The JSON for each prefix identifies the edited and queried entities.

The [experiment code and compact results](https://github.com/amirhosseinghanipour/ils-interventions) accompany the measurements. The original model weights remain available from their model distributors.

The archived/internal-replication directory contains the signed WikiText horizon summaries for all 64 held-out pairs per larger checkpoint and the source positions required for the two archive diagnostics. Two pairs with no strictly later position drop out of those diagnostics. SHA-256 fingerprints for each binary also appear in the original predictive-coverage artifact on the code branch.

The top-level 9b, 4b and 05b directories contain 128 color-training and 48 ledger-confirmation prefixes per checkpoint. NPZ files contain the source displacement x and final residual shifts y in relevant/irrelevant order. The 0.5B set includes the small-amplitude response and, on confirmation, local and training-averaged Jacobian products. The 9B screen and head-confirm files hold every screened and confirmed head intervention. The parcels subdirectories contain 48 object-packing prefixes per larger model and the second 9B component confirmation. Predictor fitting for these transfers uses the top-level color data.

The animals and carriers directories support the final frozen prototype test. Each larger model has 48 new animal prefixes. Each word in words.json has a single neutral calibration prompt, its exact tokenizer IDs, the top predicted token, and a row in vectors.npz. All prototype coefficients are learned on the earlier color data.

The blind directories contain clean dictionary and question states, the source displacement, the separate finite-effect targets and prompt records for the 64-prefix food evaluation. Only the numeric dictionaries, displacement and clean question state enter the decoder. The records and targets support evaluation and independent replay.
