"""Восстановить проверочные тексты и принятую базу из публичных закреплённых источников."""
from pathlib import Path
import argparse,hashlib,io,json,os,shutil,stat,sys,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from release import canonical,digest,file_sha256
from src.deployment import verify_artifact

p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=ROOT/'deployments/planner');args=p.parse_args();root=args.workspace.resolve()
if root.exists():raise FileExistsError('Выберите новый workspace')
config=json.loads((ROOT/'conf/release-inputs.json').read_text())
with urllib.request.urlopen(urllib.request.Request(config['url'],headers={'User-Agent':'planner-release'}),timeout=90) as response:raw=response.read(10_000_001)
if len(raw)>10_000_000 or hashlib.sha256(raw).hexdigest()!=config['sha256']:raise ValueError('Не совпал SHA-256 приложения')
with zipfile.ZipFile(io.BytesIO(raw)) as z:
 if set(z.namelist())!=set(config['files']) or sum(i.file_size for i in z.infolist())>20_000_000:raise ValueError('Неверный состав архива')
 for info in z.infolist():
  target=root/info.filename
  if not target.resolve().is_relative_to(root) or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Небезопасный путь')
 root.mkdir(parents=True)
 z.extractall(root)
m=json.loads((root/'manifest-baseline.json').read_text())
from huggingface_hub import snapshot_download
source=Path(snapshot_download(m['runtime']['base_name'],revision=m['runtime']['base_revision']))
artifact=root/m['artifact_dir'];artifact.mkdir(parents=True)
for name in m['file_hashes']:
 target=artifact/name
 if name=='chat_template.jinja':
  token_config=json.loads((source/'tokenizer_config.json').read_text());target.write_text(token_config['chat_template'])
 else:
  target.parent.mkdir(parents=True,exist_ok=True)
  try:os.link((source/name).resolve(),target)
  except OSError:shutil.copy2(source/name,target)
verify_artifact(root,m)
from src.release_bridge import verify_bound_report
verify_bound_report(root,m,json.loads((root/'release-report-baseline.json').read_text()))
print('Текстовые доказательства и принятая база восстановлены; хеши файлов и отчёт совпали.')
print('Необслуживаемые веса отклонённого адаптера в приложение не включены.')
