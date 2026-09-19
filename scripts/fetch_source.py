"""Загрузить открытый источник; проверить оба SHA-256 до установки файлов."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    meta = json.loads((ROOT / 'conf/source.json').read_text())
    target = ROOT / 'data/source'
    data = target / 'massive-ru-RU.jsonl'
    if data.exists() and hashlib.sha256(data.read_bytes()).hexdigest() == meta['file_sha256'] and (target / 'LICENSE').exists():
        print('Источник уже на диске; SHA-256 совпадает')
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / 'source.tar.gz'
        with urlopen(meta['url'], timeout=60) as response, archive.open('wb') as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != meta['archive_sha256']:
            raise ValueError('SHA-256 архива изменился; источник не принят')
        with tarfile.open(archive) as tf:
            blobs = {dest: tf.extractfile(name).read() for name, dest in meta['members'].items()}
        if hashlib.sha256(blobs['massive-ru-RU.jsonl']).hexdigest() != meta['file_sha256']:
            raise ValueError('SHA-256 русского источника не совпадает')
        target.mkdir(parents=True, exist_ok=True)
        for dest, blob in blobs.items():
            (target / dest).write_bytes(blob)
    print('Источник загружен, SHA-256 архива и JSONL проверены')


if __name__ == '__main__':
    main()
