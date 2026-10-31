import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ASSETS = Path(r"D:\DA\TTNT_De19_Assets")
os.environ.update(TTNT_APPS_DATA_DIR=str(ASSETS / 'data/apps-introductory-expanded-v4'), HF_HUB_OFFLINE='1',
                  PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', TEMP=str(ASSETS/'temp'), TMP=str(ASSETS/'temp'))
sys.path.insert(0, str(PROJECT/'scripts'))
from apps_core import canonical, digest, metrics, read, score, write
from apps_adversary import rows
from apps_model_runner import load_model, render_prompt


def prepare(root):
    cases = rows(root/'seed-42/round-1-adaptive-curriculum.jsonl')
    by_candidate = {}
    for case in cases:
        if case['curriculum_source'] == 'local_adversary':
            by_candidate.setdefault(case['candidate_id'], case)
    target = root/'curriculum-diagnosis'
    target.mkdir(exist_ok=False)
    cases = sorted(by_candidate.values(), key=lambda c:c['candidate_id'])
    path = target/'cases.jsonl'
    path.write_text(''.join(canonical(case)+'\n' for case in cases),encoding='utf-8',newline='\n')
    plan = {'scope':'In-sample Train diagnostic only; not generalization or final Test',
            'selection':'All distinct candidates used in seed42 round1; no selection by model scores',
            'case_count':len(cases),'cases_sha256':digest(path),'generation_seeds':[43,44],
            'comparison':'SFT840, matched control64, adaptive64 on identical new code prompts. No hyperparameter selection from this diagnostic.',
            'training_seed':42,'round':1,'model_profile':'2b','prompt_version':'apps-call-v2'}
    write(target/'plan.json',plan)
    shutil.copy2(Path(__file__),target/'evaluator.py')
    print('DIAGNOSTIC PLAN:', target)


def evaluate(target, variant, seed):
    import torch
    plan = read(target/'plan.json')
    if digest(target/'cases.jsonl') != plan['cases_sha256'] or seed not in plan['generation_seeds']:
        raise ValueError('Diagnostic plan mismatch')
    study = target.parent
    old = read(study/'plan.json')
    stages = read(study/'stages.json')
    adapter = Path(old['parent_adapter']) if variant=='sft' else Path(stages['s42-r1-'+variant+'-train'])/'adapter'
    output = target/(variant+'-'+str(seed))
    output.mkdir(exist_ok=False)
    cases=rows(target/'cases.jsonl')
    result={'scope':plan['scope'],'variant':variant,'seed':seed,'adapter_sha256':digest(adapter/'adapter_model.safetensors'),
            'cases_sha256':plan['cases_sha256'],'status':'running','model_test_evaluation':False}
    write(output/'result.json',result)
    model,tokenizer,eos=load_model(adapter)
    predictions=[]
    for case in cases:
        prompt=render_prompt(tokenizer,case,'apps-call-v2')
        inputs=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to('cuda:0')
        case_seed=(seed+int(hashlib.sha256(case['case_id'].encode()).hexdigest()[:8],16))%2**32
        torch.manual_seed(case_seed)
        with torch.inference_mode():
            generated=model.generate(**inputs,num_return_sequences=4,max_new_tokens=96,max_time=120,
                                     do_sample=True,temperature=1.0,top_p=0.95,top_k=20,
                                     eos_token_id=eos,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,use_cache=True)
        samples=[]
        for tokens in generated[:,inputs.input_ids.shape[1]:].tolist():
            end=next((i for i,token in enumerate(tokens) if token in eos),None)
            actual=tokens[:end+1] if end is not None else tokens
            item=score(tokenizer.decode(actual,skip_special_tokens=True).strip(),end is not None,case)
            if item.get('error')=='verifier_limit':
                raise RuntimeError('Verifier limit')
            samples.append(item)
        predictions.append({'case_id':case['case_id'],'source_id':case['source_id'],'candidate_id':case['candidate_id'],
                            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'case_seed':case_seed,'samples':samples})
        del generated,inputs
    (output/'predictions.jsonl').write_text(''.join(canonical(row)+'\n' for row in predictions),encoding='utf-8',newline='\n')
    by_id={case['case_id']:case for case in cases}
    for row in predictions:
        for sample in row['samples']:
            checked=score(sample['raw_output'],sample['terminated'],by_id[row['case_id']])
            if any(checked[key]!=sample[key] for key in ('valid_format','correct_expected','bug_detected','reward')):
                raise ValueError('Rescore mismatch')
    result.update(status='completed',metrics=metrics(predictions),predictions_sha256=digest(output/'predictions.jsonl'))
    write(output/'result.json',result)
    print(variant,seed,result['metrics'])


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.argv[1]=='--prepare':
        prepare(Path(sys.argv[2]))
    else:
        evaluate(Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]))
