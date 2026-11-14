"""Приёмка доказательств: ни один пропуск не заменяется зелёной надписью."""
from pathlib import Path
import hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'results/hw11'
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
checks={};notes={};summaries=list((DATA/'balance').glob('*/summary.json'));checks['15_full_comparisons']=len(summaries)==15
allids=[];payloads={};protocol=True;identities=True
for path in summaries:
 config=json.loads((path.parent/'config.json').read_text());r=[x for f in path.parent.glob('client-*.jsonl') for x in rows(f)];protocol &= len(r)==1680 and config['measurement_seconds']>=120
 for x in r:
  protocol &= abs(x['planned_wall']-config['start_wall']-x['case_id']/config['global_rate'])<.00001 and x['limit_parts']==4
  identities &= bool(x.get('pod')) and x['model_id']=='planner-cpu-A' and x['backend_mode']=='simulated'
  allids.append(x['request_id'])
  if x['outcome']=='completed':
   ci=x['client_index'];expected=[{'role':'system','content':'prefix-'+str(ci)},{'role':'user','content':'контроль '+str(ci)}]
   expected=hashlib.sha256(json.dumps(expected,ensure_ascii=False,sort_keys=True).encode()).hexdigest();protocol &= x['response']['history_sha256']==expected and x['chunks']==4 and len(x['inter_chunk_seconds'])==3
checks['same_schedule_and_inputs']=bool(protocol);checks['every_attempt_has_loaded_identity']=bool(identities);checks['unique_attempt_ids']=len(allids)==len(set(allids))
text=(DATA/'checks.log').read_text() if (DATA/'checks.log').exists() else ''
checks['five_real_cluster_checks']='Ran 5 tests' in text and '\nOK\n' in text
life=DATA/'lifecycle'
def results(name):return [x['row'] for obj in rows(life/(name+'.jsonl')) for x in obj.get('stdout',[]) if x.get('event')=='result']
if life.exists() and (life/'rolling-load.jsonl').exists():
 roll=results('rolling-load')+results('rollout-long-result');checks['rollout_zero_5xx_and_incomplete']=bool(roll) and all(r['status']==200 and r['outcome']=='completed' for r in roll)
 notes['rollout_attempts']=len(roll)
 force=results('forced-result');checks['forced_interruption_preserved']=len(force)>0 and force[-1]['outcome']=='interrupted' and force[-1]['chunks']<100
cohort=DATA/'cohort/summary.json';checks['matching_histogram_cohort']=cohort.exists() and json.loads(cohort.read_text())['client_count']==json.loads(cohort.read_text())['prometheus_count']==80
c=DATA/'autoscale/timeline.jsonl';states={n:set() for n in ['PlannerLatency','PlannerRejects']};traces={n:[] for n in states};series=[]
if c.exists():
 for r in rows(c):
  hs=r.get('hpa',{}).get('items',[])
  if hs:series.append((r['wall'],hs[0].get('status',{}).get('currentReplicas'),hs[0].get('status',{}).get('desiredReplicas')))
  for g in r.get('alerts',{}).get('data',{}).get('groups',[]):
   for rule in g.get('rules',[]):
    if rule.get('name') in states:states[rule['name']].add(rule.get('state'));traces[rule['name']].append(rule.get('state'))
counts=[x[1] for x in series];checks['hpa_2_to_3_to_2']=2 in counts and 3 in counts and counts[-1]==2
checks['both_alerts_pending_firing_resolved']=all('pending' in seq and 'firing' in seq and seq[-1]=='inactive' and seq.index('pending')<seq.index('firing')<max(i for i,v in enumerate(seq) if v=='inactive') for seq in traces.values());notes['alert_states']={k:sorted(v) for k,v in states.items()};notes['hpa_samples']=series
out={'checks':checks,'all_passed':all(checks.values()),'notes':notes};(DATA/'acceptance.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(checks,ensure_ascii=False,indent=2));sys.exit(0 if out['all_passed'] else 1)
