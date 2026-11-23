"""Check one prompt's quantized Qwen3.5 full and suffix runs. Run in the probe output directory."""

import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


ids = list(map(int, Path("ids.txt").read_text().split()))
inputs = {}
outputs = {}
for name in ("full", "suffix"):
    for sign in (0, 1, -1):
        path = Path(f"{name}{sign}.f32")
        inputs[path.name] = sha(path)
        array = np.fromfile(path, dtype=np.float32).reshape(len(ids), 2560)
        assert np.isfinite(array).all()
        outputs[name, sign] = array.astype(np.float64)
errors = {
    sign: float(np.abs(outputs["full", sign] - outputs["suffix", sign]).max())
    for sign in (0, 1, -1)
}
effect = (outputs["full", 1] - outputs["full", -1]) / 2
norms = np.linalg.norm(effect, axis=1)
site = len(ids) // 2
exact = all(inputs[f"full{s}.f32"] == inputs[f"suffix{s}.f32"] for s in errors)
passed = bool(exact and (norms[:site] == 0).all() and (norms[site:] > 0).all())
print(
    json.dumps(
        dict(
            passed=passed,
            model_sha256=sha(sys.argv[1]),
            suffix_sha256=sha(sys.argv[2]),
            backend="llama.cpp b12110 CPU, 4 threads, Q4_K_M",
            prompt="A scientist measured the temperature of the water at sunrise and again at noon.",
            token_ids=ids,
            source_block=15,
            target_block=31,
            source_position=site,
            donor_position=1,
            epsilon=0.05,
            source_direction="same-prompt residual at donor position minus residual at source position",
            output="final block output before normalization at all prompt positions",
            max_abs_error=errors,
            antithetic_effect_l2_by_position=norms.tolist(),
            output_sha256=inputs,
        ),
        indent=2,
        allow_nan=False,
    )
)
if not passed:
    raise SystemExit(1)
