"""Construct paired text inputs before loading either model."""
from pathlib import Path
import json
import numpy as np
from datasets import load_from_disk

here=Path(__file__).parent
config=json.loads((here/'protocol.json').read_text())
rng=np.random.default_rng(config['seed'])
data=load_from_disk('../data/wikitext-103-raw-v1')
out=Path('runs/internal-replication')
used=set()
rows=[]
for split,count in [('train',config['train_prompts']),('test',config['test_prompts'])]:
    subset=data[split]
    texts=[]
    for idx in rng.permutation(len(subset)):
        text=' '.join(subset[int(idx)]['text'].split())
        if len(text)<600 or text in used:continue
        used.add(text);texts.append((int(idx),text))
        if len(texts)==2*count:break
    assert len(texts)==2*count
    for i in range(count):
        src,donor=texts[2*i:2*i+2]
        rows.append(dict(split=split,source_row=src[0],donor_row=donor[0],source=src[1],donor=donor[1],
                         t=int(rng.integers(16,63)),u=int(rng.integers(16,63))))
(out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
with (out/'inputs.txt').open('w') as stream:
    for r in rows:
        stream.write(f"{r['t']} {r['u']}\n{r['source']}\n{r['donor']}\n")
