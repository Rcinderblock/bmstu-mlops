"""Реальные subprocess-опыты над готовым артефактом, без нового обучения."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=ROOT/'deployments/planner')
    parser.add_argument('--out',type=Path,default=ROOT/'notes/work/hw08-09-10/experiments')
    args=parser.parse_args();workspace=args.workspace.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    records=[]
    def pointer():
        path=workspace/'accepted.json';return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    def run(name,argv,success):
        before=pointer()
        cmd=[sys.executable,str(ROOT/'lab.py'),'--workspace',str(workspace),*argv]
        result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=180,
            env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PREFECT_API_URL':'http://127.0.0.1:4208/api'})
        (out/(name+'.log')).write_text(result.stdout+'\n'+result.stderr)
        row=dict(name=name,command=cmd,exit_code=result.returncode,pointer_sha256_before=before,
                 pointer_sha256_after=pointer(),flow_run_id=next((line.split()[-1] for line in result.stdout.splitlines() if line.startswith('FLOW_RUN_ID ')),None),
                 prepared='PREPARED ' in result.stdout,cache_hit='CACHE_HIT ' in result.stdout)
        records.append(row);(out/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
        assert (result.returncode==0)==success,(name,result.stdout,result.stderr)
        if not success:assert before==pointer(),name
        return row
    config=json.loads((workspace/'config.json').read_text())
    config['code_revision']=hashlib.sha256((ROOT/'src/deployment.py').read_bytes()).hexdigest()
    (workspace/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    baseline=['run','--manifest','manifest-baseline.json','--report','release-report-baseline.json']
    run('01-fail-after-prepare',baseline+['--fail-after-prepare'],False)
    resumed=run('02-resume-new-process',baseline,True);assert resumed['cache_hit']
    prior=pointer();run('03-idempotent-repeat',baseline,True);assert pointer()==prior
    events=[json.loads(x) for x in (workspace/'publications.jsonl').read_text().splitlines()];assert len(events)==1
    input_path=workspace/'input.txt';original=input_path.read_bytes()
    try:
        rows=[json.loads(x) for x in original.decode().splitlines()]
        rows[0]['input']+=' изменённая команда'
        input_path.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
        changed=run('04-change-input-same-path',baseline,False);assert changed['prepared'] and not changed['cache_hit']
    finally:input_path.write_bytes(original)
    run('05-candidate-quality-rejection',['run','--manifest','manifest-candidate.json','--report','release-report-candidate.json'],False)
    report=json.loads((workspace/'release-report-baseline.json').read_text())
    negatives={}
    negatives['incomplete']={**report,'status':'running'}
    negatives['truthy-string']={**report,'passed':'true'}
    negatives['foreign']={**report,'model_id':'foreign-model'}
    negatives['nan']=copy.deepcopy(report);negatives['nan']['metrics']['accuracy']=float('nan')
    for idx,(name,value) in enumerate(negatives.items(),6):
        filename='negative-'+name+'.json';(workspace/filename).write_text(json.dumps(value,ensure_ascii=False))
        run(f'{idx:02}-{name}',['run','--manifest','manifest-baseline.json','--report',filename],False)
    print('Все реальные опыты завершены; опубликована только исходная модель.',flush=True)


if __name__=='__main__':main()
