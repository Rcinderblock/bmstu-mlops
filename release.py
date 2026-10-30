"""Идентичность входа, полный отчёт и атомарный принятый указатель."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

HASH_FIELDS = ('model_sha256', 'protocol_sha256', 'golden_sha256', 'runtime_sha256')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def cache_key(data_path: Path, config: dict) -> str:
    return digest(canonical({'data_sha256': file_sha256(data_path), 'config': config}))


def validate_release(manifest: dict, report: dict) -> None:
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        raise ValueError('Manifest и отчёт должны быть объектами')
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('Нужна schema_version: int 1')
    for key in ('model_id', 'artifact_uri'):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError('Нет ' + key)
    for key in HASH_FIELDS:
        value = manifest.get(key)
        if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
            raise ValueError('Некорректный SHA-256: ' + key)
    runtime = manifest.get('runtime')
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError('Нет runtime')
    try:
        runtime_hash = digest(canonical(runtime))
    except (ValueError, TypeError) as exc:
        raise ValueError('Некорректный runtime') from exc
    if runtime_hash != manifest['runtime_sha256']:
        raise ValueError('Runtime изменился после оценки')
    if report.get('status') != 'complete' or report.get('passed') is not True:
        raise ValueError('Полная оценка не разрешила выпуск')
    if not isinstance(report.get('evaluation_run_id'), str) or not report['evaluation_run_id'].strip():
        raise ValueError('Нет evaluation_run_id')
    for key in ('model_id', *HASH_FIELDS):
        if report.get(key) != manifest[key]:
            raise ValueError('Отчёт не соответствует manifest: ' + key)
    metrics = report.get('metrics')
    if not isinstance(metrics, dict):
        raise ValueError('Нет metrics')
    for key in ('accuracy', 'format_rate'):
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Некорректная метрика: ' + key)


def atomic_bytes(target, raw):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.' + target.name + '-',
                                         suffix='.tmp', delete=False) as stream:
            name = stream.name
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def publish(manifest: dict, report: dict, target: Path) -> None:
    validate_release(manifest, report)
    atomic_bytes(target, canonical(manifest) + b'\n')
