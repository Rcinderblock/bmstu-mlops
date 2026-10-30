"""Быстрая приёмка без весов; при необходимости запускает свой локальный Prefect."""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


from src.prefect_local import local_prefect


def negative():
    actual=ROOT/'release.py';before=hashlib.sha256(actual.read_bytes()).hexdigest()
    tree=ast.parse(actual.read_text())
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='cache_key':
            node.body=ast.parse('return hashlib.sha256(str(data_path).encode()).hexdigest()').body
    with tempfile.TemporaryDirectory(prefix='release-negative-') as directory:
        (Path(directory)/'release.py').write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
        result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests/hw8910'),'-p','test_release.py','-v'],
            cwd=directory,env={**os.environ,'PYTHONPATH':directory})
    assert hashlib.sha256(actual.read_bytes()).hexdigest()==before,'Изменился рабочий исходник'
    if result.returncode==0:raise AssertionError('Намеренная регрессия не обнаружена')
    print('Реальные тесты отклонили копию с поломанным cache_key; рабочий файл восстановлен/не изменён.',flush=True)
    return result.returncode


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--negative',action='store_true');parser.add_argument('--assert-lightweight',action='store_true')
    args=parser.parse_args()
    if args.negative:return negative()
    if args.assert_lightweight:
        for package in ['torch','transformers','mlflow']:
            assert importlib.util.find_spec(package) is None,package+' не должен требоваться в быстром окружении'
        print('В быстром окружении нет torch, transformers, mlflow и весов.',flush=True)
    with local_prefect():
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/hw8910'),pattern='test_*.py')
        result=unittest.TextTestRunner(verbosity=2).run(suite)
    raw=(ROOT/'tests/hw8910/test_release.py').read_bytes()
    fingerprint=hashlib.sha256(b'test_release.py'+raw).hexdigest()[:12]
    assert fingerprint=='09bd65e6b886','Приложенные тесты изменились'
    print('Отпечаток приложенных тестов:',fingerprint,flush=True)
    print('Исходный test_release.py SHA256:',hashlib.sha256(raw).hexdigest(),flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':sys.exit(main())
