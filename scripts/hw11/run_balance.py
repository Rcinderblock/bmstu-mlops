"""Реальные 15×120с сравнений. Полные неудачные попытки сохраняются."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,hashlib,json,subprocess,time,sys,threading
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,KUBE,NS,k,obj,pods,wait_ready,helm
OUT=ROOT/'results/hw11/balance';OUT.mkdir(parents=True,exist_ok=True)
SCENARIO_PREFIX=''
CLIENT_SOURCE=(ROOT/'scripts/hw11/client.py').read_text()
stop=threading.Event()
def observe():
 seen={};logs=[]
 with (OUT/'timeline.jsonl').open('a',buffering=1) as stream:
  while not stop.is_set():
   row={'wall':time.time()}
   for name,arguments in [('pods',('get','pods')),('endpoints',('get','endpointslices')),('hpa',('get','hpa'))]:
    try:row[name]=obj(*arguments)
    except Exception as e:row[name]={'error':str(e)}
   stream.write(json.dumps(row)+'\n')
   for p in row.get('pods',{}).get('items',[]):
    name=p['metadata']['name']
    if p['metadata'].get('labels',{}).get('app')=='planner' and name not in seen:
     f=(OUT/(name+'.log')).open('w');proc=subprocess.Popen(['kubectl','--kubeconfig',str(KUBE),'-n',NS,'logs','-f',name],stdout=f,stderr=subprocess.STDOUT);logs.append((proc,f));seen[name]=True
   stop.wait(5)
 for proc,f in logs:proc.terminate();f.close()

def run(policy,traffic,repeat,duration,rate):
 name=f'{SCENARIO_PREFIX}{policy}-{traffic}-{repeat}';out=OUT/name;out.mkdir();start=time.time()+8;remote_out=f'/tmp/{name}-{int(start*1000)}.jsonl'
 for p in pods():
  if p['metadata'].get('deletionTimestamp') or not any(c['type']=='Ready' and c['status']=='True' for c in p['status'].get('conditions',[])):continue
  ip=p['status'].get('podIP')
  if ip:k('exec','client-0','--','python','-c',f"import urllib.request;urllib.request.urlopen(urllib.request.Request('http://{ip}:8091/admin/reset-cache',data=b'',method='POST')).read()")
 (out/'config.json').write_text(json.dumps({'policy':policy,'traffic':traffic,'repeat':repeat,'measurement_seconds':duration,'global_rate':rate,'start_wall':start,'cache_initial':'empty; both pods reset; model already warm','keepalive':False,'generator_sha256':hashlib.sha256(CLIENT_SOURCE.encode()).hexdigest(),'requests':round(duration*rate),'services':obj('get','services'),'pods':pods()},indent=2))
 processes=[]
 for ci in range(4):
  logfile=(out/f'client-{ci}.log').open('w')
  service={'ordinary':'planner','sticky':'planner-sticky','fixed':'planner-fixed'}[policy]
  args=['python','-c',CLIENT_SOURCE,'--client',str(ci),'--traffic',traffic,'--duration',str(duration),'--rate',str(rate),'--start',str(start),'--scenario',name,'--url',f'http://{service}:8091/generate','--out',remote_out]
  proc=subprocess.Popen(['kubectl','--kubeconfig',str(KUBE),'-n',NS,'exec',f'client-{ci}','--',*args],stdout=logfile,stderr=subprocess.STDOUT);processes.append((ci,proc,logfile))
 codes=[]
 for ci,proc,log in processes:
  codes.append(proc.wait());log.close()
  try:(out/f'client-{ci}.jsonl').write_text(k('exec',f'client-{ci}','--','cat',remote_out))
  except Exception as e:(out/f'client-{ci}.error').write_text(str(e))
 rows=[]
 for p in out.glob('client-*.jsonl'):rows.extend(json.loads(x) for x in p.read_text().splitlines())
 summary={'name':name,'ended_wall':time.time(),'exit_codes':codes,'expected':round(duration*rate),'sent':len(rows),'outcomes':{key:sum(r['outcome']==key for r in rows) for key in sorted({r['outcome'] for r in rows})},'by_pod':{key:sum(r['pod']==key for r in rows) for key in sorted({r['pod'] for r in rows if r['pod']})}}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 if any(codes) or len(rows)!=round(duration*rate):raise RuntimeError('Неполный прогон: сохранён, требуется разбор')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--duration',type=int,default=120);p.add_argument('--rate',type=float,default=14);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();OUT=a.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
 if any(OUT.iterdir()):raise FileExistsError('Для повторного опыта выберите новый --out; сохранённые результаты не изменены')
 helm(modelVersion='A',replicas=2,warmupSeconds=20,cacheHitCpuMs=5,cacheMissCpuMs=80,maxRunning=1,**{'hpa.enabled':False})
 wait_ready()
 (OUT/'run-environment.json').write_text(json.dumps({'backend_mode':'simulated','pods':pods(),'client_generator_sha256':hashlib.sha256(CLIENT_SOURCE.encode()).hexdigest()},indent=2))
 th=threading.Thread(target=observe);th.start()
 try:
  for traffic in ['uniform','hot']:
   for repeat in range(1,4):
    for policy in ['ordinary','sticky']:run(policy,traffic,repeat,a.duration,a.rate)
  for repeat in range(1,4):run('fixed','hot',repeat,a.duration,a.rate)
 finally:stop.set();th.join()
