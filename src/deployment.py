"""Постоянный кеш, проверка файлов и журнал выпусков одного писателя."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from release import atomic_bytes, cache_key, canonical, digest, file_sha256, publish, validate_release


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    atomic_bytes(Path(path), canonical(value) + b'\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def append_attempt(root, event, **fields):
    with (Path(root) / 'attempts.jsonl').open('a') as stream:
        stream.write(canonical(dict(event=event, at=now(), **fields)).decode() + '\n')


def preparation_config(root):
    config = read(Path(root) / 'config.json')
    return {**config, 'processing_source_sha256': file_sha256(Path(__file__))}


def preparation_key(root):
    root = Path(root)
    return cache_key(root / 'input.txt', preparation_config(root))


def prepare(root):
    root = Path(root)
    config = preparation_config(root)
    key = preparation_key(root)
    path = root / 'cache' / (key + '.json')
    if path.exists():
        value = read(path)
        if value['key'] != key or value['input_sha256'] != file_sha256(root / 'input.txt'):
            raise ValueError('Повреждён результат подготовки')
        print('CACHE_HIT', value['input_sha256'], flush=True)
        return value
    lines = (root / 'input.txt').read_text().splitlines()
    if not lines:
        raise ValueError('Пустой вход подготовки')
    value = dict(key=key, input_sha256=file_sha256(root / 'input.txt'), rows=len(lines), config=config)
    if config.get('format') == 'golden-jsonl':
        from evaluation.contract import fingerprint
        rows = [json.loads(line) for line in lines if line.strip()]
        if len(rows) < 50 or len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Golden: нужны ≥50 уникальных id')
        for row in rows:
            if (not isinstance(row.get('input'), str) or not row['input'].strip()
                    or row.get('expected') not in row.get('choices', [])
                    or not row.get('source') or not row.get('slices')):
                raise ValueError('Неверная схема проверочного вопроса')
        value['semantic_sha256'] = fingerprint(rows)
    write(path, value)
    print('PREPARED', value['input_sha256'], flush=True)
    return value


def safe_path(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Нужен относительный путь')
    path = root / relative
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError('Путь выходит за каталог артефакта')
    return path


def verify_artifact(root, manifest):
    hashes = manifest.get('file_hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Нет закреплённого списка файлов')
    base = safe_path(root, manifest.get('artifact_dir', '.'))
    actual = {}
    for name, expected in hashes.items():
        path = safe_path(base, name)
        if not path.is_file():
            raise ValueError('Нет файла артефакта: ' + name)
        actual[name] = file_sha256(path)
        if actual[name] != expected:
            raise ValueError('Файл изменился: ' + name)
    if digest(canonical(actual)) != manifest.get('model_sha256'):
        raise ValueError('Список файлов не соответствует model_sha256')
    if manifest.get('backend') == 'transformers':
        runtime = manifest['runtime']
        tokenizer_files = {name: actual[name] for name in manifest['tokenizer_files']}
        if digest(canonical(tokenizer_files)) != runtime['tokenizer_sha256']:
            raise ValueError('Не совпал токенизатор')
        if file_sha256(safe_path(base, manifest['template_file'])) != runtime['template_sha256']:
            raise ValueError('Не совпал шаблон чата')
        if manifest.get('model_kind') == 'adapter':
            base_dir = safe_path(root, manifest['base_artifact_dir'])
            base_hashes = {name: file_sha256(safe_path(base_dir, name)) for name in manifest['base_file_hashes']}
            if base_hashes != manifest['base_file_hashes']:
                raise ValueError('Изменились файлы закреплённой базы адаптера')
    return dict(model_id=manifest['model_id'], model_sha256=manifest['model_sha256'],
                artifact_uri=manifest['artifact_uri'], files=len(actual))


def assess(root, manifest, report_name, prepared):
    report = read(safe_path(root, report_name))
    if manifest.get('backend') == 'transformers':
        if prepared.get('semantic_sha256') != manifest['golden_sha256']:
            raise ValueError('Изменился golden: нужен новый отчёт оценки')
        runtime, config = manifest['runtime'], prepared['config']
        if (config['tokenizer_revision'] != runtime['base_revision']
                or config['template_sha256'] != runtime['template_sha256']
                or config['max_length'] != runtime['max_input_tokens']):
            raise ValueError('Подготовка не соответствует оценённому runtime')
        from src.release_bridge import verify_bound_report
        verify_bound_report(Path(root), manifest, report)
    elif manifest.get('backend') != 'stub' or manifest.get('synthetic') is not True or report.get('synthetic') is not True:
        raise ValueError('STUB должен быть явно синтетическим')
    report_hash = digest(canonical(report))
    write(Path(root) / 'decisions' / (report_hash + '.json'), report)
    validate_release(manifest, report)
    return dict(report=report, sha256=report_hash)


@contextmanager
def database(root):
    conn = sqlite3.connect(Path(root) / 'publications.sqlite3', timeout=10)
    try:
        conn.execute('CREATE TABLE IF NOT EXISTS releases (release_id TEXT PRIMARY KEY, model_id TEXT NOT NULL, manifest TEXT NOT NULL, state TEXT NOT NULL, published_at TEXT)')
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def project_journal(root):
    with database(root) as conn:
        rows = conn.execute("SELECT release_id,model_id,published_at FROM releases WHERE state='committed' ORDER BY published_at,release_id").fetchall()
    raw = b''.join(canonical(dict(release_id=r[0], model_id=r[1], published_at=r[2])) + b'\n' for r in rows)
    atomic_bytes(Path(root) / 'publications.jsonl', raw)


def recover(root):
    root = Path(root)
    accepted = read(root / 'accepted.json') if (root / 'accepted.json').exists() else {}
    with database(root) as conn:
        for key, payload in conn.execute("SELECT release_id,manifest FROM releases WHERE state='pending'").fetchall():
            if accepted.get('release_id') == key:
                if canonical(accepted).decode() != payload:
                    raise ValueError('Указатель не совпадает с подготовленным выпуском')
                conn.execute("UPDATE releases SET state='committed',published_at=? WHERE release_id=?", (now(), key))
                print('RECOVERED_PUBLICATION', key, flush=True)
            else:
                conn.execute("UPDATE releases SET state='aborted' WHERE release_id=?", (key,))
    project_journal(root)


def publish_once(root, manifest, decision, fail_after_pointer=False):
    root = Path(root)
    report, report_hash = decision['report'], decision['sha256']
    validate_release(manifest, report)
    if digest(canonical(report)) != report_hash:
        raise ValueError('Изменилось решение допуска')
    verify_artifact(root, manifest)
    key = digest(canonical(dict(model_id=manifest['model_id'], model_sha256=manifest['model_sha256'],
                                runtime_sha256=manifest['runtime_sha256'], decision_sha256=report_hash)))
    accepted = {**manifest, 'release_id': key,
                'decision': dict(path='decisions/' + report_hash + '.json', sha256=report_hash,
                                 evaluation_run_id=report['evaluation_run_id'])}
    with database(root) as conn:
        conn.execute("INSERT OR IGNORE INTO releases VALUES (?,?,?,'pending',NULL)", (key, manifest['model_id'], canonical(accepted).decode()))
        conn.execute("UPDATE releases SET state='pending' WHERE release_id=? AND state='aborted'", (key,))
    target = root / 'accepted.json'
    if not target.exists() or target.read_bytes() != canonical(accepted) + b'\n':
        publish(accepted, report, target)
    if fail_after_pointer:
        raise RuntimeError('Инъекция сбоя между указателем и журналом')
    with database(root) as conn:
        conn.execute("UPDATE releases SET state='committed',published_at=COALESCE(published_at,?) WHERE release_id=?", (now(), key))
    project_journal(root)
    print('PUBLISHED', manifest['model_id'], key, flush=True)
    return dict(release_id=key, model_id=manifest['model_id'], decision_sha256=report_hash)


def load_accepted(root):
    root = Path(root)
    manifest = read(root / 'accepted.json')
    if manifest.get('decision'):
        decision = manifest['decision']
        report = read(safe_path(root, decision['path']))
        if digest(canonical(report)) != decision['sha256']:
            raise ValueError('Изменилось сохранённое решение')
        if manifest.get('backend') == 'transformers':
            from src.release_bridge import verify_bound_report
            verify_bound_report(root, manifest, report)
    elif manifest.get('backend') == 'stub' and manifest.get('synthetic') is True:
        report = read(root / ('report-' + manifest['model_id'].rsplit('-', 1)[-1] + '.json'))
    else:
        raise ValueError('Нет связанного решения допуска')
    validate_release(manifest, report)
    verify_artifact(root, manifest)
    return manifest
