"""Совместимый запуск стенда и процесса Prefect; init создаёт только STUB ONLY."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import release


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))


def initialize(root):
    if root.exists():
        raise FileExistsError('Рабочий каталог уже существует. Выберите новый --workspace.')
    root.mkdir(parents=True)
    runtime={'base_revision':'fixture-only','tokenizer_sha256':'a'*64,
             'template_sha256':'b'*64,'generation':{'max_new_tokens':32,'do_sample':False},'dtype':'float32'}
    for version in ['A','B']:
        artifact=root/f'model-{version}.txt';artifact.write_text(f'STUB ONLY — version {version}\n')
        files={artifact.name:digest(artifact.read_bytes())}
        m={'schema_version':1,'model_id':f'fixture-{version}',
           'artifact_uri':artifact.name,'model_sha256':digest(canonical(files)),
           'protocol_sha256':'d'*64,'golden_sha256':'e'*64,
           'runtime_sha256':digest(canonical(runtime)),'runtime':runtime,'file_hashes':files}
        m.update(backend='stub', synthetic=True)
        report={k:m[k] for k in ['model_id','model_sha256','protocol_sha256','golden_sha256','runtime_sha256']}
        report.update(status='complete',passed=True,evaluation_run_id=f'fixture-eval-{version}',metrics={'accuracy':.8,'format_rate':1.0},synthetic=True)
        write(root/f'manifest-{version}.json',m);write(root/f'report-{version}.json',report)
    write(root/'accepted.json',read(root/'manifest-A.json'))
    write(root/'initial-accepted.json',read(root/'manifest-A.json'))
    report=read(root/'report-B.json');report['status']='running';report['passed']='false'
    write(root/'report-incomplete.json',report)
    report=read(root/'report-B.json');report['passed']=False
    write(root/'report-reject.json',report)
    write(root/'config.json',{'tokenizer_revision':'r1','template_sha256':'b'*64,'max_length':128,'code_revision':'prepare-v1'})
    (root/'input.txt').write_text('Первый пример\nВторой пример\n')
    print('Создан синтетический стенд:',root)


def prepare(root):
    from src.deployment import prepare as prepare_input
    return prepare_input(root)


def run(root, report_name, fail_after_prepare=False, manifest_name='manifest-B.json',
        fail_after_pointer=False):
    os.environ.setdefault('PREFECT_HOME', str(Path(__file__).resolve().parent / '.prefect'))
    os.environ.setdefault('PREFECT_API_URL', 'http://127.0.0.1:4208/api')
    from urllib.parse import urlparse
    if urlparse(os.environ['PREFECT_API_URL']).hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Эта работа использует локальный сервер Prefect')
    os.environ['PREFECT_SERVER_ANALYTICS_ENABLED'] = 'false'
    from src.prefect_local import local_prefect
    with local_prefect():
        from src.release_flow import release_flow
        return release_flow(str(root), manifest_name, report_name, fail_after_prepare, fail_after_pointer)


def rollback(root, manifest_name, report_name):
    from src.deployment import database
    manifest = read(root / manifest_name)
    prior = [read(root / 'initial-accepted.json')] if (root / 'initial-accepted.json').exists() else []
    with database(root) as conn:
        prior.extend(json.loads(row[0]) for row in conn.execute("SELECT manifest FROM releases WHERE state='committed'"))
    keys = ('model_id', 'model_sha256', 'runtime_sha256')
    if not any(all(old.get(k) == manifest.get(k) for k in keys) for old in prior):
        raise ValueError('Откат разрешён только к ранее принятой версии')
    return run(root, report_name, manifest_name=manifest_name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=Path('.lab'))
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('init')
    run_parser=sub.add_parser('run');run_parser.add_argument('--report',default='report-B.json');run_parser.add_argument('--fail-after-prepare',action='store_true')
    run_parser.add_argument('--manifest', default='manifest-B.json')
    run_parser.add_argument('--fail-after-pointer', action='store_true')
    back = sub.add_parser('rollback')
    back.add_argument('--manifest', required=True);back.add_argument('--report', required=True)
    args=parser.parse_args()
    if args.command=='init':initialize(args.workspace)
    elif args.command == 'rollback':
        rollback(args.workspace,args.manifest,args.report)
    else:run(args.workspace,args.report,args.fail_after_prepare,args.manifest,args.fail_after_pointer)


if __name__=='__main__':main()
