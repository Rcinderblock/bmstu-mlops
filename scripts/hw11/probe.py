"""Короткий клиент, исполняемый в клиентском Pod через kubectl exec."""
import json,sys,time,httpx
payload=json.loads(sys.argv[1]);url=sys.argv[2] if len(sys.argv)>2 else 'http://planner:8091/generate'
row={'request_id':payload['request_id'],'sent_wall':time.time(),'sent_mono':time.perf_counter(),'outcome':'interrupted','chunks':0,'status':None};first=None;chunks=[]
try:
 with httpx.Client(timeout=30,limits=httpx.Limits(max_keepalive_connections=0)) as client:
  with client.stream('POST',url,json=payload,headers={'Connection':'close'}) as r:
   row.update(status=r.status_code,pod=r.headers.get('x-pod-name'),model_id=r.headers.get('x-model-id'))
   if r.status_code!=200:row.update(outcome='rejected',error=r.read().decode())
   else:
    for line in r.iter_lines():
     if not line:continue
     part=json.loads(line)
     if part.get('done'):row.update(outcome='completed',response=part);continue
     chunks.append(part)
     if first is None:
      first=time.perf_counter();row['first_mono']=first;print(json.dumps({'event':'first','row':row}),flush=True)
except Exception as e:row['error']=type(e).__name__+': '+str(e)
row.update(complete_mono=time.perf_counter(),chunks=len(chunks));print(json.dumps({'event':'result','row':row}),flush=True)
