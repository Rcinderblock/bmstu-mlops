"""Опубликовать только маленькие текстовые входы; никаких весов/окружения."""
from pathlib import Path
import argparse,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=ROOT/'deployments/planner');args=p.parse_args();root=args.workspace.resolve()
names=['manifest-baseline.json','manifest-candidate.json','release-report-baseline.json','release-report-candidate.json','input.txt','config.json','tokenizer-equivalence.json']
names+=sorted(str(x.relative_to(root)) for x in (root/'evidence').rglob('*') if x.is_file())
assert all(Path(x).suffix not in {'.pt','.safetensors','.bin','.wav','.mp4'} for x in names)
out=ROOT/'notes/submission/hw08-09-10';out.mkdir(parents=True,exist_ok=True);target=out/'release-inputs-v1.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for name in names:
  info=zipfile.ZipInfo(name,(2026,10,30,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
  z.writestr(info,(root/name).read_bytes())
 info=zipfile.ZipInfo('LICENSE-data.txt',(2026,10,30,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
 z.writestr(info,(ROOT/'docs/MASSIVE-LICENSE.txt').read_bytes())
manifest=dict(url='https://github.com/Rcinderblock/bmstu-mlops/releases/download/release-inputs-v1/release-inputs-v1.zip',
              sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,files=names+['LICENSE-data.txt'])
(ROOT/'conf/release-inputs.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
