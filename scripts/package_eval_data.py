"""Детерминированное небольшое приложение с данными проверки, без весов."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]
FILES=['data/eval-source/annotations.json','data/eval/golden.jsonl',
       'data/eval/behavioral.jsonl','data/eval/training-index.jsonl',
       'data/eval/source-and-leakage.json','data/eval/nearest-train-review.json',
       'docs/MASSIVE-LICENSE.txt']


def main():
    out=ROOT/'notes/submission/hw06-07/hw67-evaluation-data-v1.zip'
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in FILES:
            info=zipfile.ZipInfo(name,date_time=(2000,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(ROOT/name).read_bytes())
        z.writestr(zipfile.ZipInfo('README.txt',date_time=(2000,1,1,0,0,0)),
                   'MLOps HW6+7: 54 authored golden cases, 5 behavioral cases, full training text index.\n'
                   'Source training questions: Amazon MASSIVE 1.0, ru-RU, CC BY 4.0 (included license).\n'
                   'Authored cases and semantic annotations are agent-assisted; human review is not claimed.\n'
                   'No training tensors, model weights, voice, lecture materials or secrets.\n')
    manifest=dict(url='https://github.com/Rcinderblock/bmstu-mlops/releases/download/hw67-eval-v1/'+out.name,
                  sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size,files=FILES)
    (ROOT/'conf/evaluation-data.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out,manifest['sha256'],manifest['bytes'])


if __name__=='__main__':main()
