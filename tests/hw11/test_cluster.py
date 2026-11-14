"""Все пять проверок действительно меняют/запрашивают отдельный учебный кластер."""
from pathlib import Path
import json,subprocess,sys,time,unittest,threading,uuid,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/hw11'))
from common import k,obj,pods,helm,wait_ready,KUBE,NS
OUT=Path(os.environ.get('HW11_OUT',str(ROOT/'notes/work/hw11/repeats'/time.strftime('%Y%m%d-%H%M%S')/'lifecycle'))).resolve();OUT.mkdir(parents=True,exist_ok=True)
if any(OUT.iterdir()):raise FileExistsError('HW11_OUT должен быть новым каталогом')
PROBE=(ROOT/'scripts/hw11/probe.py').read_text()
stop=threading.Event()
def snapshot():
 seen=set();followers=[]
 with (OUT/'timeline.jsonl').open('a',buffering=1) as f:
  while not stop.is_set():
   row={'wall':time.time(),'pods':obj('get','pods'),'endpoints':obj('get','endpointslices')};f.write(json.dumps(row)+'\n')
   for pod in row['pods']['items']:
    if pod['metadata'].get('labels',{}).get('app')!='planner' or pod['status'].get('phase')!='Running':continue
    name=pod['metadata']['name'];restart=sum(c.get('restartCount',0) for c in pod['status'].get('containerStatuses',[]));key=(name,restart)
    if key in seen:continue
    log=(OUT/(name+'-r'+str(restart)+'.log')).open('w');proc=subprocess.Popen(['kubectl','--kubeconfig',str(KUBE),'-n',NS,'logs','-f',name],stdout=log,stderr=subprocess.STDOUT);followers.append((proc,log));seen.add(key)
   stop.wait(1)
 for proc,log in followers:
  proc.terminate()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
  log.close()
def record(name,value):
 with (OUT/(name+'.jsonl')).open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
def payload(text='контроль',parts=4,chunk_ms=25,messages=None):return dict(request_id=str(uuid.uuid4()),session_id='review',messages=messages or [{'role':'user','content':text}],parts=parts,chunk_ms=chunk_ms)
def start(p,url='http://planner:8091/generate'):
 proc=subprocess.Popen(['kubectl','--kubeconfig',str(KUBE),'-n',NS,'exec','-i','client-0','--','python','-c',PROBE,json.dumps(p,ensure_ascii=False),url],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 return proc
def finish(proc,name):
 out,err=proc.communicate(timeout=60);rows=[json.loads(x) for x in out.splitlines() if x.startswith('{')];record(name,{'stdout':rows,'stderr':err,'returncode':proc.returncode})
 return next(r['row'] for r in reversed(rows) if r['event']=='result')
def request(name,p=None,url='http://planner:8091/generate'):return finish(start(p or payload(),url),name)
def loaded():return [p for p in pods() if p['status'].get('podIP')]
def direct(path,ip,method='GET'):
 code=f"import urllib.request,json; r=urllib.request.urlopen(urllib.request.Request('http://{ip}:8091{path}',method='{method}',data=b'' if '{method}'=='POST' else None));print(r.read().decode())"
 return json.loads(k('exec','client-0','--','python','-c',code))
class KubernetesIntegration(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.thread=threading.Thread(target=snapshot);cls.thread.start()
 @classmethod
 def tearDownClass(cls):stop.set();cls.thread.join()
 def test_01_warmup_defect_and_fixed_endpoints(self):
  k('scale','deployment/planner','--replicas=0');k('wait','--for=delete','pod','-l','app=planner','--timeout=80s')
  helm(modelVersion='A',brokenProbes=True,warmupSeconds=20);wait_ready()
  bad=[request('broken-warmup') for _ in range(8)]
  self.assertTrue(any(r['status']==503 for r in bad),'Исходный дефект не воспроизведён')
  time.sleep(21) # Дефектные старые Pod должны закончить прогрев до сравнительного исправления.
  helm(modelVersion='A',brokenProbes=False,warmupSeconds=20)
  # Во время нового прогрева рабочий Service содержит только старые прогретые Pod.
  started=time.time();observed=False;safe=[]
  while time.time()-started<24:
   current=loaded();slices=obj('get','endpointslices','-l','kubernetes.io/service-name=planner')['items']
   eligible={ip for sl in slices for e in sl.get('endpoints',[]) if e.get('conditions',{}).get('ready') for ip in e.get('addresses',[])}
   for pod in current:
    ready=any(c['type']=='Ready' and c['status']=='True' for c in pod['status'].get('conditions',[]))
    if not ready:
     observed=True;self.assertNotIn(pod['status']['podIP'],eligible)
   safe.append(request('fixed-warmup'));time.sleep(.3)
  wait_ready();self.assertTrue(observed);self.assertTrue(all(r['status']==200 for r in safe))
 def test_02_rollout_and_rollback_actual_loaded_version(self):
  # A long stream and a short control request fit without artificial queue refusal.
  helm(modelVersion='A',maxRunning=2);wait_ready()
  long=start(payload(parts=100,chunk_ms=500));first=json.loads(long.stdout.readline());record('rollout-long-first',first)
  self.assertEqual(first['row']['model_id'],'planner-cpu-A')
  helm(modelVersion='B',warmupSeconds=20,maxRunning=2)
  live=[]
  deadline=time.time()+100
  while time.time()<deadline:
   live.append(request('rolling-load'))
   if all(p['spec']['volumes'][0].get('configMap',{}).get('name')=='planner-model-b' for p in pods()) and all(any(c['type']=='Ready' and c['status']=='True' for c in p['status'].get('conditions',[])) for p in pods()):break
  wait_ready();longrow=finish(long,'rollout-long-result');self.assertEqual(longrow['outcome'],'completed');self.assertEqual(longrow['chunks'],100)
  self.assertTrue(all(r['status']==200 and r['outcome']=='completed' for r in live));self.assertEqual(request('after-good-B')['model_id'],'planner-cpu-B')
  helm(modelVersion='A');wait_ready()
  # Заведомо неготовая новая B не должна попасть в endpoints.
  helm(modelVersion='B',neverReady=True,warmupSeconds=3)
  time.sleep(12);rows=[request('bad-B-excluded') for _ in range(12)]
  self.assertTrue(all(r['model_id']=='planner-cpu-A' and r['status']==200 for r in rows))
  helm(modelVersion='A',neverReady=False);wait_ready();self.assertEqual(request('rollback-A')['model_id'],'planner-cpu-A')
 def test_03_replacement_restores_artifact_and_dialog(self):
  history=[{'role':'user','content':'диалог продолжение контроль оранжевый'},{'role':'assistant','content':'сохранённый ответ'},{'role':'user','content':'продолжай оранжевый'}]
  before=request('before-replacement',payload(messages=history));victim=before['pod'];old_uid=next(p['metadata']['uid'] for p in pods() if p['metadata']['name']==victim)
  k('delete','pod',victim);wait_ready()
  self.assertNotIn(old_uid,[p['metadata']['uid'] for p in pods()]);after=[request('after-replacement',payload(messages=history)) for _ in range(12)]
  self.assertTrue(any(r['pod']!=victim for r in after))
  for row in after:
   self.assertEqual(row['response']['artifact_sha256'],before['response']['artifact_sha256']);self.assertEqual(row['response']['response'],before['response']['response'])
 def test_04_rejections_and_forced_interruption_accounted(self):
  code="""import asyncio,json,sys,time,httpx
sys.path.insert(0,'/app/scripts/hw11');from client import attempt
async def run():
 async with httpx.AsyncClient(timeout=15,limits=httpx.Limits(max_connections=100,max_keepalive_connections=0)) as c:
  p={'session_id':'burst','messages':[{'role':'user','content':'burst'}],'parts':4,'chunk_ms':100}
  rows=await asyncio.gather(*[attempt(c,'http://planner:8091/generate',dict(p,request_id='burst-'+str(i)),time.time(),i,'accounting',0) for i in range(90)])
  print(json.dumps(rows))
asyncio.run(run())"""
  rows=json.loads(k('exec','client-0','--','python','-c',code));record('burst-all-attempts',rows);self.assertEqual(len(rows),90);self.assertTrue(any(r['status']==429 for r in rows))
  long=start(payload(parts=100,chunk_ms=200));first=json.loads(long.stdout.readline());record('forced-first',first);victim=first['row']['pod']
  cid=next(p['status']['containerStatuses'][0]['containerID'].split('://',1)[1] for p in pods() if p['metadata']['name']==victim)
  record('forced-runtime-kill',{'pod':victim,'container_id':cid,'signal':'SIGKILL','wall':time.time()})
  subprocess.run(['docker','exec','mlops-hw11-control-plane','ctr','--namespace','k8s.io','tasks','kill','--signal','SIGKILL',cid],check=True)
  broken=finish(long,'forced-result');self.assertEqual(broken['outcome'],'interrupted');self.assertLess(broken['chunks'],100)
  # rollout status alone may still contain the pre-crash status. Verify the
  # new container identity and actual backend warmup before asserting recovery.
  restored=False;deadline=time.time()+90
  while time.time()<deadline:
   victim_pod=next((p for p in pods() if p['metadata']['name']==victim),None)
   if victim_pod:
    new_cid=victim_pod['status'].get('containerStatuses',[{}])[0].get('containerID','')
    try:backend_state=direct('/debug',victim_pod['status']['podIP'])
    except subprocess.CalledProcessError:backend_state={'ready':False}
    record('forced-recovery-state',{'wall':time.time(),'container_id':new_cid,'state':backend_state})
    if new_cid and new_cid!='containerd://'+cid and backend_state.get('ready'):
     restored=True;break
   time.sleep(1)
  self.assertTrue(restored,'New container did not reload and warm up')
  wait_ready();self.assertEqual(request('forced-recovered')['outcome'],'completed')
 def test_05_two_dialogs_are_isolated(self):
  a=payload(messages=[{'role':'user','content':'секрет диалога альфа'},{'role':'user','content':'только альфа'}]);b=payload(messages=[{'role':'user','content':'секрет диалога бета'},{'role':'user','content':'только бета'}])
  a['session_id']='dialog-alpha';b['session_id']='dialog-beta'
  for _ in range(5):
   a['request_id']=str(uuid.uuid4());b['request_id']=str(uuid.uuid4())
   ra=request('dialog-alpha',a);rb=request('dialog-beta',b)
   self.assertEqual(ra['response']['response']['last_user'],'только альфа');self.assertEqual(rb['response']['response']['last_user'],'только бета');self.assertNotEqual(ra['response']['response']['history_sha256'],rb['response']['response']['history_sha256'])
if __name__=='__main__':unittest.main(verbosity=2)
