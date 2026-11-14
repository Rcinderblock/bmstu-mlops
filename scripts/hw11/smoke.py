from pathlib import Path
import json,sys,uuid
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import k,ROOT
payload={'request_id':str(uuid.uuid4()),'session_id':'video-overview','messages':[{'role':'user','content':'проверка завершённого стенда'}],'parts':4,'chunk_ms':25}
raw=k('exec','client-0','--','python','-c',(ROOT/'scripts/hw11/probe.py').read_text(),json.dumps(payload,ensure_ascii=False))
row=next(json.loads(x)['row'] for x in raw.splitlines() if json.loads(x)['event']=='result')
assert row['outcome']=='completed' and row['chunks']==4
print(json.dumps(row,ensure_ascii=False,indent=2))
