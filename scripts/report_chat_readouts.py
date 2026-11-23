"""Audit the calibrated chat cohort and export all registered readout cells."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from ilslens.eval.readout_tasks import recall_report
from scripts.readouts_after_calibration import validate_readouts


LENSES = {'logit', 'ILS', 'J-reference', 'R-reference',
          'J-released-extra (200 prompts)', 'J++ (label-tuned)'}
TASKS = {'multihop', 'multilingual', 'association', 'typo', 'poetry'}
TASKS.update({task + ':simplified' for task in TASKS - {'multilingual'}})


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_rows(rows, fields, expected):
    indexed = {tuple(row[key] for key in fields): row for row in rows}
    require(len(indexed) == len(rows) and indexed.keys() == set(expected),
            f'Missing, extra or duplicate rows for {fields}')
    return indexed


def valid_ids(ids, maximum):
    return len(set(ids)) == len(ids) <= maximum and all(type(i) is int and 0 <= i < 151936 for i in ids)


def audit(result, details, gates, items, maps):
    validate_readouts(result, True, TASKS)
    require(set(result['metadata']['label_tuned']) == LENSES, 'Incorrect lens bank')
    require(all(row['label_tuned'] == (row['lens'] == 'J++ (label-tuned)') for row in result['rows']),
            'Mislabeled summary row')
    require((result['metadata']['epsilon'], result['metadata']['gamma']) == (0.05, 0.9),
            'Readout setting changed')
    require(result['ranking'] == 'score-descending-id-ascending-v2'
            and result['sparse_pursuit_protocol'] == 'actual-marginal-gain-v2', 'Scoring protocol changed')
    retained = [r['item'] for r in gates if r['retained'] and r['item']['split'] == 'test']
    require({r['item']['task'] for r in gates} == TASKS, 'Gate families changed')
    require([(r['id'], r['family'], r['targets']) for r in items] ==
            [(r['id'], r['task'], r['intermediates']) for r in retained], 'Export mismatch')
    require(len(items) == result['retained_test_items'] == 9, 'Retained cohort changed')
    ids = [r['id'] for r in items]
    layers = range(24)
    eligible = unique_rows(maps, ('lens', 'layer'), itertools.product(LENSES, layers))
    keys = list(itertools.product(LENSES, ids, layers))
    top = unique_rows([r for r in details if 'fdr_selected_ids' in r],
                      ('lens', 'item_id', 'layer'), keys)
    pursuit = unique_rows([r for r in details if 'pursuit_ids' in r],
                          ('lens', 'item_id', 'layer'), keys)
    require(len(details) == 2 * len(keys), 'Unexpected detail records')
    for key in keys:
        t, p = top[key], pursuit[key]
        require(t['label_tuned'] == p['label_tuned'] == (key[0] == 'J++ (label-tuned)'), 'Tuning label')
        require(all(valid_ids(t[field], 10) for field in ('raw_top10', 'z_top10'))
                and valid_ids(t['fdr_selected_ids'], 200), 'Invalid calibrated IDs')
        coeff = p['pursuit_coefficients']
        require(valid_ids(p['pursuit_ids'], 25) and len(coeff) == len(p['pursuit_ids'])
                and all(math.isfinite(c) and c > 0 for c in coeff)
                and 0 <= p['pursuit_residual_fraction'] <= 1.000001, 'Invalid sparse pursuit')
    recalls = {(r['lens'], r['task'], r['layer'], r['mode'], r['k']): r
               for r in result['rows'] if 'k' in r}
    dead = unique_rows([r for r in result['rows'] if 'k' not in r],
                       ('lens', 'task', 'layer'), itertools.product(LENSES, TASKS, layers))
    for name, task in itertools.product(LENSES, TASKS):
        cohort = [r for r in items if r['family'] == task]
        size = max((len(r['targets']) for r in cohort), default=1)
        for mode in ('raw', 'z'):
            ranks = np.full((len(cohort), 24, size), np.nan)
            for layer in layers:
                rates = []
                for i, item in enumerate(cohort):
                    t = top[name, item['id'], layer]
                    ranked = t[mode + '_top10']
                    rates.append(t[mode + '_dead_top10_fraction'])
                    for j, target in enumerate(item['targets']):
                        candidates = eligible[name, layer][mode][target]
                        if candidates:
                            # Rank 11 denotes any scoreable miss for both registered cutoffs.
                            ranks[i, layer, j] = min((k for k, token in enumerate(ranked, 1)
                                                      if token in candidates), default=11)
                field = mode + '_dead_top10_fraction'
                rates = [rate for rate in rates if rate is not None]
                require(dead[name, task, layer][field] == (float(np.mean(rates)) if rates else None),
                        f'Dead rate mismatch at {(name, task, layer, mode)}')
            for row in recall_report(ranks, [r['id'] for r in cohort]):
                key = (name, task, row['layer'], mode, row['k'])
                require(all(recalls[key][f] == v for f, v in row.items()), f'Recall mismatch at {key}')
    return Counter(r['family'] for r in items)


def calibration_evidence(items):
    """Bind token eligibility to each saved map, using the original training-frequency mask."""
    import torch
    import yaml
    from transformers import AutoTokenizer
    from ilslens.calibration import validate_calibration_cache
    from ilslens.eval.readout_tasks import candidate_ids
    from ilslens.jpp import readout_filter

    root = Path('runs/calibration')
    timing = read(root / 'timing.json')
    config = yaml.safe_load(Path('configs/calibration.yaml').read_text())
    tokenizer = AutoTokenizer.from_pretrained(os.environ['ILS_MODEL_PATH'], local_files_only=True)
    concepts = {target for item in items for target in item['targets']}
    tokens = {target: candidate_ids(tokenizer, target) for target in sorted(concepts)}
    excluded = readout_filter(tokenizer, 151936)
    rows = []
    for entry in timing['fitted_maps'] + timing['reused_maps']:
        name, layer = entry['lens'], entry['layer']
        path = root / name / f'layer-{layer:02d}.pt'
        calibration = torch.load(path, weights_only=True, map_location='cpu')
        validate_calibration_cache(calibration, entry['input_fingerprint'], 100000, 151936)
        metadata = calibration['metadata']
        expected = dict(lens=name, layer=layer, label_tuned=name.startswith('J++'),
                        reference_split='test', calibration_config=config)
        require(all(metadata[k] == v for k, v in expected.items()), f'Calibration mismatch at {path}')
        raw = {target: [i for i in ids if not (name.startswith('J++') and excluded[i])]
               for target, ids in tokens.items()}
        z = {target: [i for i in ids if not calibration['dead_mask'][i] and calibration['records'][i]['std'] > 0]
             for target, ids in raw.items()}
        rows.append(dict(lens=name, layer=layer, input_fingerprint=entry['input_fingerprint'],
                         sha256=sha256(path), raw=raw, z=z))
        del calibration
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/readouts-chat'))
    parser.add_argument('--maps', type=Path, help='Reuse previously verified map evidence instead of reloading all 144 maps')
    parser.add_argument('--output', type=Path, default=Path('artifacts/qwen-chat-readouts'))
    args = parser.parse_args()
    paths = {key: args.run / (key + '.json') for key in ('results', 'details', 'gates', 'evaluation-items')}
    result, details, gates, items = (read(path) for path in paths.values())
    expected_gate_hash = read(Path('artifacts/qwen-capability-chat-audit.json'))['source_sha256']
    require(sha256(paths['gates']) == expected_gate_hash, 'Gate file differs from audited cohort')
    maps = read(args.maps) if args.maps else calibration_evidence(items)
    counts = audit(result, details, gates, items, maps)
    args.output.mkdir(parents=True, exist_ok=True)
    map_path = args.output / 'verified-maps.json'
    map_path.write_text(json.dumps(maps, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    fields = ['lens', 'label_tuned', 'task', 'retained_test_items', 'layer', 'mode', 'k', 'recall',
              'ci_low', 'ci_high', 'n', 'unscoreable_items', 'raw_dead_top10_fraction', 'z_dead_top10_fraction']
    table = args.output / 'metrics.csv'
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source in result['rows']:
            row = dict(source, retained_test_items=counts[source['task']])
            if 'ci' in row:
                row['ci_low'], row['ci_high'] = row.pop('ci')
            writer.writerow(row)
    exported = args.output / 'evaluation-items.json'
    exported.write_text(json.dumps(items, indent=2, allow_nan=False) + '\n')
    evidence = args.output / 'item-metrics.csv'
    family = {item['id']: item['family'] for item in items}
    item_rows = {}
    for record in details:
        key = (record['lens'], record['item_id'], record['layer'])
        item_rows.setdefault(key, {'task': family[record['item_id']]}).update(record)
    detail_fields = ['lens', 'label_tuned', 'task', 'item_id', 'layer', 'raw_top10', 'z_top10',
                     'raw_dead_top10_fraction', 'z_dead_top10_fraction', 'fdr_selected_ids',
                     'pursuit_ids', 'pursuit_coefficients', 'pursuit_residual_fraction']
    with evidence.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=detail_fields)
        writer.writeheader()
        for key in sorted(item_rows):
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v
                             for k, v in item_rows[key].items()})
    provenance_paths = [
        Path('configs/readout-chat.yaml'), Path('configs/readout-simple.json'),
        Path('runs/qwen-proposal-v4/manifest.json'), Path('artifacts/qwen-primary/results.json'),
        Path('runs/faithfulness-v1/manifest.json'), Path('runs/faithfulness-v1/results.json'),
        Path('runs/jpp-chat-v1/J++-label-tuned.pt'), Path('runs/jpp-chat-v1/metadata.json'),
        Path('runs/corpus-frequencies.pt'), Path('runs/calibration/activation-manifest.json'),
        Path('runs/calibration/timing.json'), Path('scripts/evaluate_readouts.py'),
        Path('scripts/report_chat_readouts.py'),
    ]
    manifest = dict(input_sha256={str(p): sha256(p) for p in provenance_paths},
                    source_sha256={str(p): sha256(p) for p in paths.values()},
                    published_sha256={p.name: sha256(p) for p in (map_path, table, evidence, exported)},
                    protocol=result['prompt_protocol'], metadata=result['metadata'], ranking=result['ranking'],
                    sparse_pursuit_protocol=result['sparse_pursuit_protocol'],
                    recall_cells=sum('k' in r for r in result['rows']), dead_rate_rows=sum('k' not in r for r in result['rows']),
                    item_layer_lens_cells=len(details)//2, retained_test_items=dict(counts))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f"Verified {manifest['recall_cells']} recall cells and {manifest['item_layer_lens_cells']} item/layer/lens cells")


if __name__ == '__main__':
    main()
