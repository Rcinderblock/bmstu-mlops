from pathlib import Path
import argparse,json,time,threading,urllib.request,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import k,obj,helm,wait_ready,pods,ROOT
import run_balance as load
OUT=ROOT/'results/hw11/autoscale';OUT.mkdir(parents=True,exist_ok=True)
stop=threading.Event()
def get(url):
 try:
  with urllib.request.urlopen(url,timeout=5) as r:return json.load(r)
 except Exception as exc:return {'error':str(exc)}
def observe():
 with (OUT/'timeline.jsonl').open('a',buffering=1) as f:
  while not stop.is_set():
   row={'wall':time.time(),'hpa':obj('get','hpa'),'pods':obj('get','pods'),'alerts':get('http://127.0.0.1:19090/api/v1/rules')}
   try:row['resource_metrics']=json.loads(k('get','--raw','/apis/metrics.k8s.io/v1beta1/namespaces/mlops-hw11/pods'))
   except Exception as e:row['resource_metrics_error']=str(e)
   f.write(json.dumps(row)+'\n');stop.wait(5)
def wait_minimum(name):
 start=time.time()
 while time.time()-start<180:
  h=obj('get','hpa','planner');n=h.get('status',{}).get('currentReplicas');desired=h.get('status',{}).get('desiredReplicas')
  if n==2 and desired==2:
   (OUT/(name+'-downscale.json')).write_text(json.dumps({'observed_wall':time.time(),'seconds_since_load_finished':time.time()-start,'hpa':h},indent=2));return
  time.sleep(5)
 raise RuntimeError('HPA не вернулся к минимуму: сохранён фактический журнал')
def main():
 global OUT
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);args=p.parse_args();OUT=args.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
 if any(OUT.iterdir()):raise FileExistsError('Для повторного опыта выберите новый --out')
 load.OUT=OUT;load.SCENARIO_PREFIX='c-'
 helm(modelVersion='A',cacheHitCpuMs=80,cacheMissCpuMs=80);wait_ready();time.sleep(20)
 helm(modelVersion='A',cacheHitCpuMs=80,cacheMissCpuMs=80,**{'hpa.enabled':True,'hpa.target':65})
 k('scale','deployment/planner','--replicas=2');wait_ready()
 (OUT/'initial.json').write_text(json.dumps({'wall':time.time(),'pods':pods(),'hpa':obj('get','hpa','planner')},indent=2))
 th=threading.Thread(target=observe);th.start()
 try:
  load.run('ordinary','uniform',1,120,30)
  wait_minimum('uniform')
  load.run('sticky','hot',1,120,14)
  wait_minimum('hot')
 finally:stop.set();th.join()
 h=obj('get','hpa','planner');(OUT/'hpa-final.json').write_text(json.dumps(h,indent=2))
 with (OUT/'timeline.jsonl').open('a') as f:f.write(json.dumps({'wall':time.time(),'hpa':{'items':[h]},'pods':obj('get','pods'),'alerts':get('http://127.0.0.1:19090/api/v1/rules'),'final':True})+'\n')
if __name__=='__main__':main()
