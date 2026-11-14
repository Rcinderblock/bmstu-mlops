from pathlib import Path
import json,subprocess,time,os
ROOT=Path(__file__).resolve().parents[2]
KUBE=ROOT/'notes/work/hw11/kubeconfig'
HELM=ROOT/'notes/work/hw11/tools/helm'
NS='mlops-hw11'
def cmd(args,**kw):return subprocess.check_output([str(x) for x in args],text=True,**kw)
def k(*args):return cmd(['kubectl','--kubeconfig',KUBE,'-n',NS,*args])
def obj(*args):return json.loads(k(*args,'-o','json'))
def helm(**values):
 args=[HELM,'upgrade','--install','planner',ROOT/'infra/helm/planner','--kubeconfig',KUBE,'-n',NS,'--create-namespace']
 for key,value in values.items():args+=['--set',key+'='+(str(value).lower() if isinstance(value,bool) else str(value))]
 return cmd(args)
def pods():return obj('get','pods','-l','app=planner')['items']
def wait_ready(timeout=150):return k('rollout','status','deploy/planner','--timeout='+str(timeout)+'s')
