from pathlib import Path
import yaml,json
root=Path(__file__).resolve().parents[2];ns='mlops-hw11';docs=[]
def add(x):docs.append(x)
add({'apiVersion':'v1','kind':'ServiceAccount','metadata':{'name':'prometheus','namespace':ns}})
add({'apiVersion':'rbac.authorization.k8s.io/v1','kind':'Role','metadata':{'name':'prometheus','namespace':ns},'rules':[{'apiGroups':[''],'resources':['pods'],'verbs':['get','list','watch']}]})
add({'apiVersion':'rbac.authorization.k8s.io/v1','kind':'RoleBinding','metadata':{'name':'prometheus','namespace':ns},'subjects':[{'kind':'ServiceAccount','name':'prometheus','namespace':ns}],'roleRef':{'apiGroup':'rbac.authorization.k8s.io','kind':'Role','name':'prometheus'}})
add({'apiVersion':'v1','kind':'ConfigMap','metadata':{'name':'prometheus-config','namespace':ns},'data':{n:(root/'infra/monitoring'/n).read_text() for n in ['prometheus.yaml','alerts.yaml']}})
add({'apiVersion':'v1','kind':'ConfigMap','metadata':{'name':'grafana-config','namespace':ns},'data':{'datasource.yaml':yaml.safe_dump({'apiVersion':1,'datasources':[{'name':'Prometheus','uid':'prometheus','type':'prometheus','access':'proxy','url':'http://prometheus:9090','isDefault':True}]}),'dashboard.yaml':yaml.safe_dump({'apiVersion':1,'providers':[{'name':'planner','type':'file','options':{'path':'/dashboards'}}]}),'dashboard.json':(root/'infra/monitoring/dashboard.json').read_text()}})
for name,image,port,container in [
 ('prometheus','prom/prometheus:v3.5.0',9090,{'args':['--config.file=/etc/prometheus/prometheus.yaml','--storage.tsdb.path=/prometheus','--storage.tsdb.retention.time=2h','--storage.tsdb.retention.size=128MB'],'volumeMounts':[{'name':'config','mountPath':'/etc/prometheus'},{'name':'storage','mountPath':'/prometheus'}]}),
 ('grafana','grafana/grafana:12.2.0',3000,{'env':[{'name':'GF_AUTH_ANONYMOUS_ENABLED','value':'true'},{'name':'GF_AUTH_ANONYMOUS_ORG_ROLE','value':'Viewer'},{'name':'GF_ANALYTICS_REPORTING_ENABLED','value':'false'},{'name':'GF_ANALYTICS_CHECK_FOR_UPDATES','value':'false'}],'volumeMounts':[{'name':'config','mountPath':'/etc/grafana/provisioning/datasources/datasource.yaml','subPath':'datasource.yaml'},{'name':'config','mountPath':'/etc/grafana/provisioning/dashboards/dashboard.yaml','subPath':'dashboard.yaml'},{'name':'config','mountPath':'/dashboards/dashboard.json','subPath':'dashboard.json'},{'name':'storage','mountPath':'/var/lib/grafana'}]})]:
 container.update(name=name,image=image,ports=[{'containerPort':port}],resources={'requests':{'cpu':'100m','memory':'128Mi'},'limits':{'cpu':'500m','memory':'256Mi'}})
 spec={'securityContext':{'fsGroup':65534 if name=='prometheus' else 472},'containers':[container],'volumes':[{'name':'config','configMap':{'name':name+'-config'}},{'name':'storage','emptyDir':{'sizeLimit':'256Mi' if name=='prometheus' else '64Mi'}}]}
 if name=='prometheus':spec['serviceAccountName']='prometheus'
 add({'apiVersion':'apps/v1','kind':'Deployment','metadata':{'name':name,'namespace':ns},'spec':{'replicas':1,'selector':{'matchLabels':{'app':name}},'template':{'metadata':{'labels':{'app':name}},'spec':spec}}})
 add({'apiVersion':'v1','kind':'Service','metadata':{'name':name,'namespace':ns},'spec':{'selector':{'app':name},'ports':[{'port':port}]}})
(root/'infra/monitoring/stack.yaml').write_text(yaml.safe_dump_all(docs,sort_keys=False,allow_unicode=True))
