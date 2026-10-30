"""Настоящий HTTP, прогрев, 8+8 попыток и перезапуск принятой модели."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import uuid
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=ROOT/'deployments/planner')
    parser.add_argument('--out',type=Path,default=ROOT/'notes/work/hw08-09-10/api')
    parser.add_argument('--port',type=int,default=8091)
    args=parser.parse_args();args.workspace=args.workspace.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'lifecycle.json').exists() or (args.out/'attempts.jsonl').exists():parser.error('Для нового опыта укажите новый --out; старые попытки сохраняются')
    with socket.socket() as sock:sock.bind(('127.0.0.1',args.port))
    url=f'http://127.0.0.1:{args.port}'
    lifecycle=[];attempts=[];process=None;logs=[]
    def request(path,payload=None,timeout=120):
        started=time.perf_counter();status=0;body=None;error=None
        try:
            raw=None if payload is None else json.dumps(payload,ensure_ascii=False).encode()
            req=Request(url+path,data=raw,headers={'Content-Type':'application/json'})
            try:
                with urlopen(req,timeout=timeout) as response:status=response.status;body=json.load(response)
            except HTTPError as exc:status=exc.code;body=json.load(exc);error='HTTP '+str(status)
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        return dict(path=path,status=status,error=error,response=body,client_seconds=time.perf_counter()-started)
    def save_lifecycle(event,**fields):
        row=dict(event=event,**fields);lifecycle.append(row)
        (args.out/'lifecycle.json').write_text(json.dumps(lifecycle,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    def stop():
        nonlocal process
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        process=None
    def start(label):
        nonlocal process
        log=(args.out/(label+'-server.log')).open('w');logs.append(log)
        command=[sys.executable,str(ROOT/'service.py'),'--workspace',str(args.workspace),'--port',str(args.port),'--warmup-seconds','2','--debug']
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
        deadline=time.monotonic()+180;health_seen=False;not_ready_seen=False
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Сервер завершился: '+str(log.name))
            health=request('/health',timeout=2)
            if health['status']==200:
                if not health_seen:save_lifecycle(label+'-health',pid=process.pid,**health);health_seen=True
                ready=request('/ready',timeout=2)
                if ready['status']==503 and not not_ready_seen:
                    save_lifecycle(label+'-not-ready',**ready);not_ready_seen=True
                if ready['status']==200:
                    save_lifecycle(label+'-ready',pid=process.pid,**ready)
                    assert not_ready_seen,'Не наблюдалась стадия прогрева'
                    return ready['response']
                debug=request('/debug',timeout=2)
                if debug['status']==200 and debug['response'].get('load_error'):
                    raise RuntimeError(debug['response']['load_error'])
            time.sleep(.15)
        raise TimeoutError('Модель не стала готова')
    try:
        accepted=json.loads((args.workspace/'accepted.json').read_text())
        ready=start('initial')
        assert ready['model_id']==accepted['model_id'] and ready['runtime_sha256']==accepted['runtime_sha256']
        before=request('/debug')['response']['calls']
        for payload in [{'prompt':'','max_new_tokens':32},{'prompt':'покажи будильники','max_new_tokens':100000},{'prompt':'покажи будильники','max_new_tokens':'32'},{'prompt':'покажи будильники','max_new_tokens':1}]:
            result=request('/generate',payload);save_lifecycle('invalid-request',payload=payload,**result)
            assert 400<=result['status']<500,result
        after=request('/debug')['response']['calls'];assert before==after
        save_lifecycle('invalid-requests-did-not-compute',calls_before=before,calls_after=after)
        config=json.loads((ROOT/'conf/api-load.json').read_text())
        smoke=request('/generate',dict(prompt=config['requests'][0]['prompt'],max_new_tokens=config['max_new_tokens']))
        save_lifecycle('real-smoke',**smoke);assert smoke['status']==200 and smoke['response']['backend']=='transformers'
        batches=[]
        for concurrency in config['concurrency']:
            started=time.perf_counter()
            def attempt(case):
                row=request('/generate',dict(prompt=case['prompt'],max_new_tokens=config['max_new_tokens']))
                body=row['response'] if isinstance(row['response'],dict) else {}
                return dict(attempt_id=str(uuid.uuid4()),case_id=case['id'],prompt=case['prompt'],
                            model_id=body.get('model_id'),runtime_sha256=body.get('runtime_sha256'),request_id=body.get('request_id'),
                            generation_seconds=body.get('generation_seconds'),output_tokens=body.get('output_tokens'),
                            concurrency=concurrency,max_new_tokens=config['max_new_tokens'],**row)
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures=[pool.submit(attempt,case) for case in config['requests']]
                for future in as_completed(futures):
                    row=future.result();attempts.append(row)
                    with (args.out/'attempts.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                    print('ATTEMPT',row['case_id'],'concurrency',concurrency,'status',row['status'],flush=True)
            seconds=time.perf_counter()-started
            current=[r for r in attempts if r['concurrency']==concurrency]
            success=[r for r in current if r['status']==200]
            errors=[r for r in current if r['status']!=200]
            batch=dict(concurrency=concurrency,n=len(current),successes=len(success),errors=len(errors),batch_seconds=seconds,
                       successes_per_second=len(success)/seconds,
                       mean_success_latency=statistics.mean(r['client_seconds'] for r in success) if success else None,
                       max_success_latency=max((r['client_seconds'] for r in success),default=None),
                       mean_error_latency=statistics.mean(r['client_seconds'] for r in errors) if errors else None,
                       max_error_latency=max((r['client_seconds'] for r in errors),default=None),
                       mean_queue_seconds=statistics.mean(r['response']['queue_seconds'] for r in success) if success else None,
                       mean_generation_seconds=statistics.mean(r['response']['generation_seconds'] for r in success) if success else None,
                       output_tokens=sum(r['response']['output_tokens'] for r in success))
            batches.append(batch);(args.out/'summary.json').write_text(json.dumps(batches,ensure_ascii=False,indent=2)+'\n')
            print('BATCH',json.dumps(batch,ensure_ascii=False),flush=True)
        stop();restarted=start('restart')
        result=request('/generate',dict(prompt=config['requests'][0]['prompt'],max_new_tokens=config['max_new_tokens']))
        save_lifecycle('restart-real-smoke',**result)
        assert restarted['model_id']==accepted['model_id'] and result['response']['runtime_sha256']==accepted['runtime_sha256']
        assert result['status']==200
        assert len(attempts)==16
        assert all(r['response']['model_id']==accepted['model_id'] for r in attempts if r['status']==200)
        assert all(row['successes']==8 and row['errors']==0 for row in batches),'Ошибки сохранены; нужно расследовать перед сдачей'
    finally:
        stop()
        for log in logs:log.close()


if __name__=='__main__':main()
