"""Скачать опубликованное приложение и проверить SHA-256 до записи данных."""
from pathlib import Path
import hashlib,io,json,stat,urllib.request,zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    cfg=json.loads((ROOT/'conf/evaluation-data.json').read_text())
    req=urllib.request.Request(cfg['url'],headers={'User-Agent':'bmstu-mlops-coursework'})
    with urllib.request.urlopen(req,timeout=90) as response:raw=response.read(10_000_001)
    if len(raw)>10_000_000 or hashlib.sha256(raw).hexdigest()!=cfg['sha256']:
        raise ValueError('Архив не совпадает с закреплённой версией')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if set(z.namelist())!=set(cfg['files'])|{'README.txt'}:
            raise ValueError('Неожиданные файлы в архиве')
        if sum(p.file_size for p in z.infolist())>20_000_000:
            raise ValueError('Слишком большой распакованный архив')
        for info in z.infolist():
            if info.filename=='README.txt':continue
            path=ROOT/info.filename
            if not path.resolve().is_relative_to(ROOT.resolve()) or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError('Недопустимый путь в архиве')
            path.parent.mkdir(parents=True,exist_ok=True)
            # Лицензия уже в Git; сверяем, а не заменяем исходник.
            data=z.read(info)
            if info.filename.startswith('docs/'):
                if path.read_bytes()!=data:raise ValueError('Лицензия не совпадает')
            else:path.write_bytes(data)
    print('Данные проверки восстановлены; SHA-256 совпал')


if __name__=='__main__':main()
