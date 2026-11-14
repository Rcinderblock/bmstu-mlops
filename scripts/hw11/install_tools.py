"""Официальные закреплённые kind/Helm; проверка SHA-256 до исполнения."""
from pathlib import Path
import hashlib,io,platform,tarfile,urllib.request
root=Path(__file__).resolve().parents[2]/'notes/work/hw11/tools';root.mkdir(parents=True,exist_ok=True)
system=platform.system().lower();machine=platform.machine().lower();arch='arm64' if machine in ['aarch64','arm64'] else 'amd64'
if system not in ['darwin','linux']:raise RuntimeError('В этом скрипте поддержаны macOS/Linux')
def get(url):
 with urllib.request.urlopen(url,timeout=90) as r:return r.read()
def verified(url):
 raw=get(url);expected=get(url+'.sha256sum').decode().split()[0]
 if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('SHA-256 mismatch: '+url)
 return raw
raw=verified(f'https://github.com/kubernetes-sigs/kind/releases/download/v0.30.0/kind-{system}-{arch}')
p=root/'kind';p.write_bytes(raw);p.chmod(0o755)
raw=verified(f'https://get.helm.sh/helm-v3.19.0-{system}-{arch}.tar.gz')
with tarfile.open(fileobj=io.BytesIO(raw)) as tar:raw=tar.extractfile(f'{system}-{arch}/helm').read()
p=root/'helm';p.write_bytes(raw);p.chmod(0o755)
print('Проверенные kind v0.30.0 и Helm v3.19.0:',root)
