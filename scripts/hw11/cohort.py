"""Один изолированный набор: точное совпадение client и classic bucket count."""
from pathlib import Path
import json,sys,time,urllib.request,urllib.parse,os
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import k,pods,helm,wait_ready,ROOT
OUT=Path(os.environ.get('HW11_OUT',str(ROOT/'notes/work/hw11/repeats'/time.strftime('%Y%m%d-%H%M%S')/'cohort'))).resolve();OUT.mkdir(parents=True,exist_ok=True)
if any(OUT.iterdir()):raise FileExistsError('HW11_OUT должен быть новым каталогом')
helm(modelVersion='A',warmupSeconds=3,cacheHitCpuMs=5,cacheMissCpuMs=80)
k('rollout','restart','deployment/planner');wait_ready()
current=[p for p in pods() if not p['metadata'].get('deletionTimestamp') and any(c['type']=='Ready' and c['status']=='True' for c in p['status'].get('conditions',[]))];assert len(current)==2;names=[p['metadata']['name'] for p in current];matcher='|'.join(names)
def query(q):
 with urllib.request.urlopen('http://127.0.0.1:19090/api/v1/query?'+urllib.parse.urlencode({'query':q}),timeout=10) as r:return json.load(r)
base='planner_ttfc_seconds'
selector='{pod=~"'+matcher+'"}'
time.sleep(7);start=time.time();before=query('sum('+base+'_count'+selector+')');(OUT/'before.json').write_text(json.dumps(before,indent=2))
assert float(before['data']['result'][0]['value'][1])==0,'Cohort must start at zero'
code="""import asyncio,json,sys,time,httpx
sys.path.insert(0,'/app/scripts/hw11');from client import attempt
async def run():
 rows=[]
 async with httpx.AsyncClient(timeout=15,limits=httpx.Limits(max_keepalive_connections=0)) as c:
  for i in range(80):
   p={'request_id':'cohort-'+str(i),'session_id':'cohort','messages':[{'role':'user','content':'cohort-'+str(i%4)}],'parts':4,'chunk_ms':25}
   rows.append(await attempt(c,'http://planner:8091/generate',p,time.time(),i,'histogram-cohort',0))
 print(json.dumps(rows))
asyncio.run(run())"""
rows=json.loads(k('exec','client-0','--','python','-c',code));end=time.time();(OUT/'requests.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n');assert len(rows)==80 and all(r['outcome']=='completed' for r in rows)
time.sleep(7)
queries={'count':'sum('+base+'_count'+selector+')','buckets':'sum by(le) ('+base+'_bucket'+selector+')','p95':'histogram_quantile(0.95, sum by(le) ('+base+'_bucket'+selector+'))'}
after={name:query(q) for name,q in queries.items()};(OUT/'after.json').write_text(json.dumps(after,indent=2))
assert float(after['count']['data']['result'][0]['value'][1])==80
client=sorted(r['ttfc_seconds'] for r in rows);summary={'start_wall':start,'end_wall':end,'pods':names,'client_count':80,'prometheus_count':80,'client_p95_seconds':client[75]+.05*(client[76]-client[75]),'histogram_p95_seconds':float(after['p95']['data']['result'][0]['value'][1]),'queries':queries,'explanation':'Fresh Pod counters; only this cohort generated first chunks. Client adds network overhead; classic bucket quantile interpolates.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
