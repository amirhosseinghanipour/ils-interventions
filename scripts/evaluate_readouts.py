"""Capability-filtered five-task readouts. Defaults to guarded GPU execution."""
import argparse,hashlib,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch,yaml
from ilslens.model import load_local
from ilslens.artifacts import primary_bank
from ilslens.eval.readout_tasks import load_reference_items,load_simplified_items,generate_answers,gate_item,candidate_ids,recall_report,validate_labeled_dev
from ilslens.hooks import capture_blocks
from ilslens.readout import raw,sparse_pursuit,target_ranks
from ilslens.calibration import CALIBRATION_VERSION,calibrated_readout,matrix_fingerprint
from ilslens.jpp import readout_filter
from ilslens.eval.prompting import read_site,render_read_prompt

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True)
    parser.add_argument('--run',default='runs/qwen-main');parser.add_argument('--config',default='configs/readout.yaml')
    parser.add_argument('--output',default='runs/readouts');parser.add_argument('--device',default='cuda')
    parser.add_argument('--gate-only',action='store_true');parser.add_argument('--jpp')
    parser.add_argument('--calibration-root',help='Override artifact location without changing capability-gate settings')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    calibration_root=Path(args.calibration_root or config['calibration_root'])
    prompting=dict(prompt_format=config.get('prompt_format','raw'),system_prompt=config.get('system_prompt'))
    if (config['draws'],config['required_correct'],config['temperature'],config['dev_fraction'])!=(10,8,.8,.2):
        raise ValueError('This registered protocol requires 10 draws, 8 correct, temperature .8 and 20% dev')
    (root/'estimate.json').write_text(json.dumps(config,indent=2));start=time.perf_counter()
    items=load_reference_items(args.source,seed=config['seed'])
    if config.get('limit_per_task'):
        counts={};limited=[]
        for item in items:
            counts[item.task]=counts.get(item.task,0)+1
            if counts[item.task]<=config['limit_per_task']:limited.append(item)
        items=limited
    model,tokenizer=load_local(args.device)
    gates=[]
    def evaluate_gate(index,item,bank='reference'):
        path=root/f'gate-{bank}-{index:04d}.json'
        if path.exists():
            record=json.loads(path.read_text())
            if record['item']!=asdict(item) or record.get('config')!=config:raise ValueError('Gate cache settings changed')
        else:
            answers=generate_answers(model,tokenizer,item.capability_prompt,seed=config['seed']+index*100,max_new_tokens=config['max_new_tokens'],**prompting)
            completions=answers if item.capability_prompt==item.prompt else generate_answers(model,tokenizer,item.prompt,seed=config['seed']+index*100+20,max_new_tokens=config['max_new_tokens'],**prompting)
            record=dict(gate_item(item,answers,completions,rendered_prompt=render_read_prompt(tokenizer,item.prompt,**prompting)),config=config)
            path.write_text(json.dumps(record,indent=2))
        return record
    for index,item in enumerate(items):gates.append(evaluate_gate(index,item))
    fallback_families=[]
    if config.get('simplified_bank'):
        for family in sorted({item.task for item in items}):
            retained_count=sum(r['retained'] and r['item']['split']=='test' and r['item']['task']==family for r in gates)
            if retained_count<config['minimum_reference_test_items']:fallback_families.append(family)
        supplemental=load_simplified_items(config['simplified_bank'],fallback_families,config['seed'])
        for index,item in enumerate(supplemental):gates.append(evaluate_gate(len(items)+index,item,bank='simplified'))
        items.extend(supplemental)
    (root/'gates.json').write_text(json.dumps(gates,indent=2))
    if args.gate_only:return
    bank,metadata=primary_bank(args.run,epsilon=config['epsilon'],gamma=config['gamma'])
    if args.jpp or config.get('jpp_path'):
        artifact=torch.load(args.jpp or config['jpp_path'],weights_only=True,map_location='cpu')
        if artifact['metadata'].get('label_tuned') is not True:raise ValueError('J++ must be labeled label-tuned')
        if artifact['metadata']['primary_fingerprint']!=metadata['fingerprint']:raise ValueError('J++ fitting provenance differs')
        validate_labeled_dev(gates,artifact['metadata'],hashlib.sha256((root/'gates.json').read_bytes()).hexdigest())
        bank['J++ (label-tuned)']=artifact['matrices'];metadata['label_tuned']['J++ (label-tuned)']=True
    retained=[record for record in gates if record['retained'] and record['item']['split']=='test']
    exported=[]
    for record in retained:
        item=record['item'];tokens,position=read_site(tokenizer,item['prompt'],item['task'],**prompting)
        exported.append(dict(id=item['id'],family=item['task'],prompt=item['prompt'],input_ids=tokens[0].tolist(),positions=[position],
            targets=item['intermediates'],answers=item['answers'],actual_output=record['natural_completions'][0],split='test'))
    (root/'evaluation-items.json').write_text(json.dumps(exported,indent=2))
    rows=[];missing_calibration=[];details=[]
    for name,maps in bank.items():
        excluded=readout_filter(tokenizer,model.config.vocab_size).to(model.device) if name.startswith('J++') else None
        layers=sorted(maps)
        tasks=sorted({item.task for item in items})
        selected=[record['item'] for record in retained]
        max_concepts=max((len(item['intermediates']) for item in selected),default=1)
        arrays={mode:np.full((len(selected),len(layers),max_concepts),np.nan) for mode in ('raw','z')}
        dead_rates={(task,layer):[] for task in tasks for layer in layers}
        z_dead_rates={(task,layer):[] for task in tasks for layer in layers}
        # Keep only the read-site vectors. Each calibration map serves all items before release.
        sites=[]
        for item in selected:
            tokens,position=read_site(tokenizer,item['prompt'],item['task'],**prompting)
            with torch.no_grad(),capture_blocks(model,layers) as states:model.model(tokens.to(model.device),use_cache=False)
            sites.append({layer:states[layer][0,position].clone() for layer in layers})
        for l,layer in enumerate(layers):
            calibration_path=calibration_root/name/f'layer-{layer:02d}.pt'
            calibration=None
            if calibration_path.exists():
                calibration=torch.load(calibration_path,weights_only=True,map_location='cpu')
                if calibration.get('calibration_version')!=CALIBRATION_VERSION:raise ValueError('Calibration method changed; regenerate calibration before evaluating')
                if calibration['reference_count']<100000:raise ValueError('Insufficient reference activations for production calibration')
                if calibration['metadata']['primary_fingerprint']!=metadata['fingerprint']:
                    raise ValueError('Calibration and lens fitting provenance differ')
                if calibration['metadata'].get('matrix_sha256')!=matrix_fingerprint(maps[layer]):
                    raise ValueError('Calibration belongs to a different lens matrix')
                if (calibration['metadata']['epsilon'],calibration['metadata']['gamma'])!=(config['epsilon'],config['gamma']):
                    raise ValueError('Calibration uses a different ILS epsilon or horizon')
            else:missing_calibration.append(str(calibration_path))
            for i,item in enumerate(selected):
                site=sites[i][layer]
                with torch.no_grad():scores=raw(site,maps[layer].to(model.device),model.lm_head.weight,model.model.norm.weight)
                if excluded is not None:scores=scores.masked_fill(excluded,-torch.inf)
                z=None
                if calibration is not None:
                    calibrated=calibrated_readout(scores,calibration);z=calibrated['z']
                    if calibrated['raw_dead_top10_fraction'] is not None:dead_rates[item['task'],layer].append(calibrated['raw_dead_top10_fraction'])
                    if calibrated['z_dead_top10_fraction'] is not None:z_dead_rates[item['task'],layer].append(calibrated['z_dead_top10_fraction'])
                    details.append(dict(item_id=item['id'],lens=name,label_tuned=name.startswith('J++'),layer=layer,
                        raw_top10=calibrated['raw_top10_ids'],z_top10=calibrated['z_top10_ids'],
                        fdr_selected_ids=calibrated['accepted_ids'],raw_dead_top10_fraction=calibrated['raw_dead_top10_fraction'],
                        z_dead_top10_fraction=calibrated['z_dead_top10_fraction']))
                if excluded is not None:
                    scores=scores.masked_fill(excluded,-torch.inf)
                    if z is not None:z=z.masked_fill(excluded,-torch.inf)
                if config.get('sparse_pursuit',False):
                    with torch.no_grad():
                        dictionary=model.lm_head.weight@maps[layer].to(model.device)
                        if excluded is not None:dictionary[excluded]=0
                        coefficients,residual=sparse_pursuit(site,dictionary,seed=config['seed'])
                        support=torch.where(coefficients>0)[0]
                    details.append(dict(item_id=item['id'],lens=name,label_tuned=name.startswith('J++'),layer=layer,
                        pursuit_ids=support.cpu().tolist(),pursuit_coefficients=coefficients[support].cpu().tolist(),
                        pursuit_residual_fraction=float(residual.square().sum()/site.square().sum().clamp_min(1e-30))))
                    del dictionary
                for c,concept in enumerate(item['intermediates']):
                    ids=candidate_ids(tokenizer,concept)
                    if not ids:continue
                    for mode,values in [('raw',scores),('z',z)]:
                        if values is not None:
                            arrays[mode][i,l,c]=float(target_ranks(values,ids))
            del calibration
            print(f'Completed {name} layer {layer}',flush=True)
        for task in tasks:
            indices=[i for i,item in enumerate(selected) if item['task']==task]
            for mode,array in arrays.items():
                reports=recall_report(array[indices],[selected[i]['id'] for i in indices])
                for report in reports:
                    report['layer']=layers[report['layer']] if isinstance(report['layer'],int) else report['layer']
                    rows.append(dict(report,lens=name,label_tuned=name.startswith('J++'),task=task,mode=mode))
            for layer in layers:
                rates=dead_rates[task,layer]
                rows.append(dict(lens=name,label_tuned=name.startswith('J++'),task=task,layer=layer,
                    raw_dead_top10_fraction=None if not rates else float(np.mean(rates)),
                    z_dead_top10_fraction=None if not z_dead_rates[task,layer] else float(np.mean(z_dead_rates[task,layer]))))
    (root/'details.json').write_text(json.dumps(details,indent=2))
    (root/'results.json').write_text(json.dumps(dict(rows=rows,metadata=metadata,missing_calibration=missing_calibration,
        seconds=time.perf_counter()-start,calibration_root=str(calibration_root),prompt_protocol=prompting,complete_calibration=not missing_calibration,retained_test_items=len(retained),ranking='score-descending-id-ascending-v2',sparse_pursuit_protocol='actual-marginal-gain-v2',
        simplified_families=fallback_families,simplified_bank=config.get('simplified_bank'),
        caveats=['Strict leakage exclusion may reject all poetry items','No implicit LLM-judge substitutions']),indent=2))

if __name__=='__main__':main()
