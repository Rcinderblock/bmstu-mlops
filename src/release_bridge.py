"""Связать сохранённую полную оценку с неизменными файлами и runtime API."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess

from evaluation.contract import fingerprint, read_rows, validate_golden
from evaluation.solution import quality_gate, score
from release import canonical, digest, file_sha256
from src.deployment import read, safe_path, write

PROJECT = Path(__file__).resolve().parents[1]


def evidence(root, manifest):
    for name, expected in manifest['evidence_hashes'].items():
        if file_sha256(safe_path(root, name)) != expected:
            raise ValueError('Изменился исходный отчёт/протокол: ' + name)
    source = Path(root) / 'evidence'
    config = read(source / 'params.json')
    golden, behavior, training = [read_rows(source / name) for name in ['golden.jsonl', 'behavioral.jsonl', 'training-index.jsonl']]
    validate_golden(golden, training, behavior)
    code = {p.name: p.read_text() for p in sorted((source / 'evaluation-code').glob('*.py'))}
    for name in code:
        if file_sha256(PROJECT / 'evaluation' / name) != file_sha256(source / 'evaluation-code' / name):
            raise ValueError('Изменился код оценки: ' + name)
    protocol = fingerprint(dict(config=config, golden=golden, behavioral=behavior, training_index=training, evaluation_code=code))
    baseline_spec = read(source / 'baseline-spec.json')
    reports = {role: read(source / (role + '-eval.json')) for role in ['baseline', 'candidate']}
    for role, report in reports.items():
        if (report.get('protocol_sha256') != protocol or report.get('golden_sha256') != fingerprint(golden)
                or report.get('baseline_spec_sha256') != fingerprint(baseline_spec)):
            raise ValueError('Отчёт относится к другому протоколу/набору')
        outputs = read(source / (role + '-outputs.json'))
        if report['outputs_sha256'] != fingerprint(outputs):
            raise ValueError('Изменились сырые ответы')
        by_id = {o['id']: o for o in outputs}
        if len(by_id) != len(outputs) or set(by_id) != {r['id'] for r in golden + behavior}:
            raise ValueError('Неполный или повторный набор ответов')
        recalculated = score(golden, [by_id[r['id']] for r in golden])
        if any(report[k] != recalculated[k] for k in ('overall', 'slices', 'per_item')):
            raise ValueError('Метрики не соответствуют сырым ответам')
        if report['behavioral'] != score(behavior, [by_id[r['id']] for r in behavior]):
            raise ValueError('Behavioral не соответствует сырым ответам')
    if reports['baseline']['model'] != baseline_spec:
        raise ValueError('Исходная версия была заменена')
    if reports['candidate']['model'] != read(source / 'candidate-spec.json'):
        raise ValueError('Изменился выбранный кандидат')
    runtime = manifest['runtime']
    origin = read(source / 'origin.json')
    generation = dict(system=config['system'], max_new_tokens=config['max_new_tokens'],
                      enable_thinking=config['enable_thinking'], do_sample=False, seed=config['seed'])
    if (runtime['generation'] != generation or runtime['base_revision'] != baseline_spec['revision']
            or runtime['dtype'] != origin['training_manifest']['model']['dtype']
            or runtime['device'] != origin['training_run']['tags']['device']
            or runtime['packages'] != origin['packages']):
        raise ValueError('API не соответствует runtime полной оценки')
    if manifest['protocol_sha256'] != protocol or manifest['golden_sha256'] != fingerprint(golden):
        raise ValueError('Manifest относится к другим данным оценки')
    return config, reports


def derive_report(root, manifest):
    config, reports = evidence(root, manifest)
    role = manifest['evaluation_role']
    selected = reports[role]
    try:
        decision = quality_gate(reports['baseline'], selected, config['critical_slices'])
    except ValueError as exc:
        decision = dict(passed=False, reason=str(exc))
    return dict(status='complete', passed=decision['passed'],
                evaluation_run_id=selected['evaluation_run_id'], model_id=manifest['model_id'],
                **{key: manifest[key] for key in ['model_sha256', 'protocol_sha256', 'golden_sha256', 'runtime_sha256']},
                metrics={key: selected['overall'][key] for key in ['accuracy', 'format_rate']},
                gate=decision, source_reports=manifest['evidence_hashes'],
                decision_kind='retain-evaluated-baseline' if role == 'baseline' else 'candidate-quality-gate',
                evaluation_reused=True, synthetic=False)


def verify_bound_report(root, manifest, report):
    if report != derive_report(root, manifest):
        raise ValueError('Решение не связано с полными исходными отчётами и их хешами')


def historical_code(tags):
    commit = tags['git_sha']
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=PROJECT, check=True)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, 'src'], cwd=PROJECT, text=True).splitlines()
    h = hashlib.sha256()
    for name in sorted(n for n in names if n.endswith('.py') and n.count('/') == 1):
        h.update(Path(name).name.encode())
        h.update(subprocess.check_output(['git', 'show', commit + ':' + name], cwd=PROJECT))
    if h.hexdigest() != tags['code_sha256']:
        raise ValueError('Код исторического training run не совпал с Git')
    return dict(commit=commit, code_sha256=h.hexdigest(), verified_against_historical_commit=True)


def copy_files(source, dest, names):
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        p = source / name
        target = dest / name
        if not target.exists():
            try:
                os.link(p.resolve(), target)
            except OSError:
                shutil.copy2(p, target)


def initialize(root, reports_dir=None):
    root = Path(root).resolve()
    reports_dir = Path(reports_dir or PROJECT / 'metrics')
    if root.exists():
        raise FileExistsError('Выберите новый workspace; текущий не перезаписывается')
    from evaluation.cli import manifest_for, tracking
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    spec = read(PROJECT / 'metrics/candidate-model.json')
    training_manifest, adapter = manifest_for(spec)  # именно сохранённая версия, без нового sweep
    _, client = tracking()
    run = client.get_run(spec['run_id'])
    code_proof = historical_code(run.data.tags)
    requirements = (adapter.parents[1] / 'requirements.txt').read_text().splitlines()
    packages = {}
    for name in ['torch', 'transformers', 'peft']:
        packages[name] = next(line.split('==', 1)[1] for line in requirements if line.startswith(name + '=='))
        if version(name) != packages[name]:
            raise ValueError('Нужен прежний runtime или новая оценка: ' + name)
    baseline_spec = read(PROJECT / 'evaluation/baseline.json')
    base = Path(snapshot_download(baseline_spec['name'], revision=baseline_spec['revision'], local_files_only=True))
    root.mkdir(parents=True)
    source = root / 'evidence';source.mkdir()
    for role in ['baseline', 'candidate']:
        for kind in ['eval', 'outputs']:
            shutil.copy2(reports_dir / f'{role}-{kind}.json', source / f'{role}-{kind}.json')
    for name in ['golden.jsonl', 'behavioral.jsonl', 'training-index.jsonl']:
        shutil.copy2(PROJECT / 'data/eval' / name, source / name)
    shutil.copy2(PROJECT / 'evaluation/params.json', source / 'params.json')
    shutil.copy2(PROJECT / 'evaluation/baseline.json', source / 'baseline-spec.json')
    shutil.copy2(PROJECT / 'metrics/candidate-model.json', source / 'candidate-spec.json')
    (source / 'evaluation-code').mkdir()
    for p in sorted((PROJECT / 'evaluation').glob('*.py')):
        shutil.copy2(p, source / 'evaluation-code' / p.name)
    origin = dict(training_manifest=training_manifest, training_run=dict(run_id=run.info.run_id, tags=run.data.tags),
                  packages=packages, historical_code=code_proof,
                  operation='Read an existing training artifact; no training was executed')
    write(source / 'origin.json', origin)
    base_names = sorted(p.name for p in base.iterdir() if p.is_file())
    base_out = root / 'artifacts/base';copy_files(base, base_out, base_names)
    old_tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    (base_out / 'chat_template.jinja').write_text(old_tokenizer.chat_template)
    adapter_names = sorted(p.name for p in adapter.iterdir() if p.is_file())
    copy_files(adapter, root / 'artifacts/candidate', adapter_names)
    hashes = {str(p.relative_to(root)): file_sha256(p) for p in sorted(source.rglob('*')) if p.is_file()}
    config = read(source / 'params.json')
    reports = {role: read(source / f'{role}-eval.json') for role in ['baseline', 'candidate']}
    for role, folder in [('baseline', base_out), ('candidate', root / 'artifacts/candidate')]:
        files = {p.name: file_sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}
        token_names = sorted(n for n in files if n in ['tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt', 'chat_template.jinja'])
        runtime = dict(base_name=baseline_spec['name'], base_revision=baseline_spec['revision'],
                       dtype=training_manifest['model']['dtype'], device=run.data.tags['device'],
                       tokenizer_sha256=digest(canonical({n: files[n] for n in token_names})),
                       template_sha256=files['chat_template.jinja'], max_input_tokens=4096, packages=packages,
                       generation=dict(system=config['system'], max_new_tokens=config['max_new_tokens'],
                                       do_sample=False, seed=config['seed'], enable_thinking=config['enable_thinking']))
        manifest = dict(schema_version=1, model_id=('planner-baseline-' + baseline_spec['revision'][:12]) if role == 'baseline' else ('planner-adapter-' + spec['run_id']),
                        artifact_uri=('hf://' + baseline_spec['name'] + '@' + baseline_spec['revision']) if role == 'baseline' else spec['uri'],
                        backend='transformers', model_kind='base' if role == 'baseline' else 'adapter',
                        artifact_dir=str(folder.relative_to(root)), base_artifact_dir='artifacts/base', file_hashes=files,
                        tokenizer_files=token_names, template_file='chat_template.jinja', runtime=runtime,
                        model_sha256=digest(canonical(files)), runtime_sha256=digest(canonical(runtime)),
                        protocol_sha256=reports[role]['protocol_sha256'], golden_sha256=reports[role]['golden_sha256'],
                        evaluation_role=role, source_training_run_id=spec['run_id'],
                        source_evaluation_run_id=reports[role]['evaluation_run_id'], evidence_hashes=hashes)
        if role == 'candidate':
            manifest['base_file_hashes'] = read(root / 'manifest-baseline.json')['file_hashes']
        write(root / ('manifest-' + role + '.json'), manifest)
        write(root / ('release-report-' + role + '.json'), derive_report(root, manifest))
    shutil.copy2(source / 'golden.jsonl', root / 'input.txt')
    write(root / 'config.json', dict(format='golden-jsonl', tokenizer_revision=baseline_spec['revision'],
                                    template_sha256=read(root / 'manifest-baseline.json')['runtime']['template_sha256'],
                                    max_length=4096, code_revision=file_sha256(PROJECT / 'src/deployment.py')))
    # Доказываем, что экспорт шаблона не изменил реальные входные токены оценки.
    exported = AutoTokenizer.from_pretrained(str(base_out), local_files_only=True)
    cases = read_rows(source / 'golden.jsonl') + read_rows(source / 'behavioral.jsonl')
    for row in cases:
        messages = [{'role': 'system', 'content': config['system']}, {'role': 'user', 'content': row['input']}]
        kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=config['enable_thinking'])
        a, b = old_tokenizer.apply_chat_template(messages, **kwargs), exported.apply_chat_template(messages, **kwargs)
        if a != b or old_tokenizer(a)['input_ids'] != exported(b)['input_ids']:
            raise ValueError('Изменилось построение входа; нужна новая оценка')
    write(root / 'tokenizer-equivalence.json', dict(cases=len(cases), identical_chat_text=True, identical_token_ids=True))
    print('Готовы manifest и связанные отчёты. Новое обучение/генерация не выполнялись.', flush=True)
    print('baseline gate:', read(root / 'release-report-baseline.json')['passed'],
          'candidate gate:', read(root / 'release-report-candidate.json')['passed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('deployments/planner'))
    parser.add_argument('--reports-dir', type=Path, default=Path('metrics'))
    args = parser.parse_args()
    initialize(args.workspace, args.reports_dir)
