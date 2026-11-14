"""Реальные 15×120с сравнений. Полные неудачные попытки сохраняются."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,json,subprocess,time,sys,threading
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,KUBE,NS,k,obj,pods,wait_ready,helm
OUT=ROOT/'results/hw11/balance';OUT.mkdir(parents=True,exist_ok=True)
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
 name=f'{policy}-{traffic}-{repeat}';out=OUT/name;out.mkdir();start=time.time()+8
 for p in pods():
  ip=p['status'].get('podIP')
  if ip:k('exec','client-0','--','python','-c',f"import urllib.request;urllib.request.urlopen(urllib.request.Request('http://{ip}:8091/admin/reset-cache',data=b'',method='POST')).read()")
 (out/'config.json').write_text(json.dumps({'policy':policy,'traffic':traffic,'repeat':repeat,'measurement_seconds':duration,'global_rate':rate,'start_wall':start,'cache_initial':'empty; both pods reset; model already warm','keepalive':False,'requests':round(duration*rate),'services':obj('get','services'),'pods':pods()},indent=2))
 processes=[]
 for ci in range(4):
  logfile=(out/f'client-{ci}.log').open('w')
  service={'ordinary':'planner','sticky':'planner-sticky','fixed':'planner-fixed'}[policy]
  args=['python','/app/scripts/hw11/client.py','--client',str(ci),'--traffic',traffic,'--duration',str(duration),'--rate',str(rate),'--start',str(start),'--scenario',name,'--url',f'http://{service}:8091/generate','--out',f'/tmp/{name}.jsonl']
  proc=subprocess.Popen(['kubectl','--kubeconfig',str(KUBE),'-n',NS,'exec',f'client-{ci}','--',*args],stdout=logfile,stderr=subprocess.STDOUT);processes.append((ci,proc,logfile))
 codes=[]
 for ci,proc,log in processes:
  codes.append(proc.wait());log.close()
  try:(out/f'client-{ci}.jsonl').write_text(k('exec',f'client-{ci}','--','cat',f'/tmp/{name}.jsonl'))
  except Exception as e:(out/f'client-{ci}.error').write_text(str(e))
 rows=[]
 for p in out.glob('client-*.jsonl'):rows.extend(json.loads(x) for x in p.read_text().splitlines())
 summary={'name':name,'ended_wall':time.time(),'exit_codes':codes,'expected':round(duration*rate),'sent':len(rows),'outcomes':{key:sum(r['outcome']==key for r in rows) for key in sorted({r['outcome'] for r in rows})},'by_pod':{key:sum(r['pod']==key for r in rows) for key in sorted({r['pod'] for r in rows if r['pod']})}}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 if any(codes) or len(rows)!=round(duration*rate):raise RuntimeError('Неполный прогон: сохранён, требуется разбор')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--duration',type=int,default=120);p.add_argument('--rate',type=float,default=14);a=p.parse_args()
 th=threading.Thread(target=observe);th.start()
 try:
  for traffic in ['uniform','hot']:
   for repeat in range(1,4):
    for policy in ['ordinary','sticky']:run(policy,traffic,repeat,a.duration,a.rate)
  for repeat in range(1,4):run('fixed','hot',repeat,a.duration,a.rate)
 finally:stop.set();th.join()
