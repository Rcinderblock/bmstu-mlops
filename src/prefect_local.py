"""Локальный Prefect: использовать работающий сервер или запустить свой на время."""
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def local_prefect():
    os.environ['PREFECT_API_URL'] = 'http://127.0.0.1:4208/api'
    os.environ['PREFECT_SERVER_ANALYTICS_ENABLED'] = 'false'
    os.environ['PREFECT_HOME'] = str(ROOT / '.prefect')

    def healthy():
        try:
            with urlopen('http://127.0.0.1:4208/api/health', timeout=1) as response:
                return response.status == 200
        except Exception:
            return False
    proc, log = None, None
    try:
        if not healthy():
            path = ROOT / '.prefect'
            path.mkdir(exist_ok=True)
            log = (path / 'local-server.log').open('w')
            proc = subprocess.Popen([sys.executable, '-m', 'prefect', 'server', 'start',
                                     '--host', '127.0.0.1', '--port', '4208'],
                                    stdout=log, stderr=subprocess.STDOUT, env=os.environ)
            for _ in range(300):
                if healthy():
                    break
                if proc.poll() is not None:
                    raise RuntimeError('Prefect не запустился; см. .prefect/local-server.log')
                time.sleep(.1)
            if not healthy():
                raise TimeoutError('Prefect не стал готов за 30 секунд')
        yield
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if log is not None:
            log.close()
