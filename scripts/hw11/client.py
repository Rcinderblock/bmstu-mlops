"""Четыре открытых генератора. Каждая строка — отдельная HTTP-попытка, без повторов."""
import argparse,asyncio,json,os,time,socket
from pathlib import Path
import httpx

async def attempt(client,url,payload,planned,case_id,scenario,client_index):
    sent=time.perf_counter();wall=time.time();parts=[];first=None;last=None;intervals=[];done=None
    row=dict(case_id=case_id,scenario=scenario,client_index=client_index,client_ip=socket.gethostbyname(socket.gethostname()),request_id=payload['request_id'],planned_wall=planned,planned_offset=sent-wall+planned,sent_wall=wall,sent_mono=sent,send_lag_seconds=wall-planned,input_chars=sum(len(x['content']) for x in payload['messages']),limit_parts=payload['parts'],backend_mode='simulated',status=None,pod=None,model_id=None)
    try:
        async with client.stream('POST',url,json=payload,headers={'Connection':'close'}) as r:
            row.update(status=r.status_code,pod=r.headers.get('x-pod-name'),model_id=r.headers.get('x-model-id'))
            if r.status_code!=200:
                await r.aread();row['outcome']='rejected'
            else:
                async for line in r.aiter_lines():
                    if not line:continue
                    obj=json.loads(line);now=time.perf_counter()
                    if obj.get('done'):done=obj;continue
                    if first is None:first=now
                    if last is not None:intervals.append(now-last)
                    last=now;parts.append(obj)
                row['outcome']='completed' if done and len(parts)==payload['parts'] else 'interrupted'
                if done:row.update(queue_seconds=done['queue_seconds'],cache_hit=done['cache_hit'],response=done['response'],artifact_sha256=done['artifact_sha256'],server_duration=done['service_seconds'])
    except httpx.TimeoutException as exc:row.update(outcome='timed_out',error=type(exc).__name__)
    except Exception as exc:row.update(outcome='interrupted',error=type(exc).__name__+': '+str(exc))
    end=time.perf_counter();row.update(first_chunk_mono=first,complete_mono=end,ttfc_seconds=None if first is None else first-sent,duration_seconds=end-sent,chunks=len(parts),inter_chunk_seconds=intervals)
    return row

async def main(args):
    start=args.start or time.time()+2
    rows=[];tasks=[]
    async with httpx.AsyncClient(timeout=httpx.Timeout(15,connect=3),limits=httpx.Limits(max_connections=100,max_keepalive_connections=0)) as client:
        for i in range(round(args.duration*args.rate)):
            ci=i%4 if args.traffic=='uniform' else (0 if i%10<8 else 1+(i//10+i%2)%3)
            if ci!=args.client:continue
            planned=start+i/args.rate
            await asyncio.sleep(max(0,planned-time.time()))
            text='контроль '+str(ci)
            payload=dict(request_id=f'{args.scenario}-{i}',session_id=f'dialog-{ci}',messages=[{'role':'system','content':'prefix-'+str(ci)},{'role':'user','content':text}],parts=args.parts,chunk_ms=args.chunk_ms)
            tasks.append(asyncio.create_task(attempt(client,args.url,payload,planned,i,args.scenario,ci)))
        for task in asyncio.as_completed(tasks):
            row=await task;rows.append(row)
            with open(args.out,'a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(dict(scenario=args.scenario,client=args.client,planned=len(tasks),rows=len(rows),outcomes={k:sum(r['outcome']==k for r in rows) for k in sorted({r['outcome'] for r in rows})})),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--url',default='http://planner:8091/generate');p.add_argument('--client',type=int,required=True);p.add_argument('--traffic',choices=['uniform','hot'],default='uniform');p.add_argument('--duration',type=float,default=120);p.add_argument('--rate',type=float,default=14);p.add_argument('--start',type=float);p.add_argument('--scenario',required=True);p.add_argument('--parts',type=int,default=4);p.add_argument('--chunk-ms',type=int,default=25);p.add_argument('--out',default='/tmp/requests.jsonl');asyncio.run(main(p.parse_args()))
