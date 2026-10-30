"""Одинаковые HTTP/процессные опыты над исходным и исправленным кодом."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT = Path(os.environ.get('RELEASE_TEST_ROOT', Path(__file__).resolve().parents[2])).resolve()


class IncidentIntegrationTests(unittest.TestCase):
    def record(self, event, **fields):
        location = os.environ.get('RELEASE_TEST_EVIDENCE')
        if location:
            path = Path(location) / (self._testMethodName + '.jsonl')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as stream:
                stream.write(json.dumps(dict(event=event, **fields), ensure_ascii=False) + '\n')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'workspace'
        self.lab('init')

    def lab(self, *args):
        pointer = self.root / 'accepted.json'
        before = hashlib.sha256(pointer.read_bytes()).hexdigest() if pointer.exists() else None
        result = subprocess.run(
            [sys.executable, str(PROJECT / 'lab.py'), '--workspace', str(self.root), *args],
            cwd=self.tmp.name, text=True, capture_output=True, timeout=150,
            env={**os.environ, 'PYTHONPATH': str(PROJECT)},
        )
        self.record('subprocess', arguments=list(args), returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr, pointer_sha256_before=before,
                    pointer_sha256_after=hashlib.sha256(pointer.read_bytes()).hexdigest() if pointer.exists() else None)
        if args == ('init',):
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def call(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(self.url + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=5) as response:
                body = json.load(response)
                self.record('http', path=path, payload=payload, status=response.status, response=body)
                return response.status, body
        except HTTPError as error:
            body = json.load(error)
            self.record('http', path=path, payload=payload, status=error.code, response=body)
            return error.code, body

    def server(self, seconds=0):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        log = open(Path(self.tmp.name) / f'http-{time.monotonic_ns()}.log', 'w+')
        self.addCleanup(log.close)
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT / 'service.py'), '--workspace', str(self.root),
             '--port', str(port), '--warmup-seconds', str(seconds)],
            cwd=self.tmp.name, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONPATH': str(PROJECT)},
        )

        def stop():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait(timeout=5)
        self.addCleanup(stop)
        self.url = f'http://127.0.0.1:{port}'
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log.seek(0)
                self.fail('HTTP-процесс завершился до запуска: ' + log.read())
            try:
                if self.call('/health')[0] == 200:
                    return proc
            except (URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(.05)
        self.fail('HTTP-сервер не начал отвечать')

    def wait_ready(self):
        for _ in range(200):
            if self.call('/ready')[0] == 200:
                return
            time.sleep(.03)
        self.fail('Прогрев не завершился')

    def test_cache_recovery_and_changed_bytes_in_new_process(self):
        failed = self.lab('run', '--fail-after-prepare')
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn('PREPARED', failed.stdout + failed.stderr)
        replay = self.lab('run')
        self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
        self.assertIn('CACHE_HIT', replay.stdout + replay.stderr)
        (self.root / 'input.txt').write_text('Новые байты по тому же пути\n')
        changed = self.lab('run')
        self.assertEqual(changed.returncode, 0, changed.stdout + changed.stderr)
        self.assertIn('PREPARED', changed.stdout + changed.stderr)
        self.assertNotIn('CACHE_HIT', changed.stdout + changed.stderr)

    def test_incomplete_and_foreign_reports_preserve_pointer(self):
        before = (self.root / 'accepted.json').read_bytes()
        for report in ['report-incomplete.json', 'report-A.json', 'report-reject.json']:
            with self.subTest(report=report):
                result = self.lab('run', '--report', report)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual((self.root / 'accepted.json').read_bytes(), before)

    def test_identical_publication_and_repair_after_pointer_crash(self):
        for _ in range(2):
            result = self.lab('run')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        events = (self.root / 'publications.jsonl').read_text().splitlines()
        self.assertEqual(len(events), 1, 'Один выпуск B размножился в разных процессах')
        # Новый workspace нужен, чтобы повтор уже принятого B не скрыл точку сбоя.
        import shutil
        shutil.rmtree(self.root)
        self.lab('init')
        failed = self.lab('run', '--fail-after-pointer')
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(json.loads((self.root / 'accepted.json').read_text())['model_id'], 'fixture-B')
        restored = self.lab('run')
        self.assertEqual(restored.returncode, 0, restored.stdout + restored.stderr)
        events = [json.loads(line) for line in (self.root / 'publications.jsonl').read_text().splitlines()]
        self.record('recovered-journal', events=events)
        self.assertEqual([e['model_id'] for e in events], ['fixture-B'])

    def test_atomic_failure_preserves_real_file_bytes(self):
        source = '''
import json
from pathlib import Path
from unittest.mock import patch
import release
root=Path(__import__('sys').argv[1]); target=root/'accepted.json'; before=target.read_bytes()
m=json.loads((root/'manifest-B.json').read_text());r=json.loads((root/'report-B.json').read_text())
with patch('os.replace',side_effect=OSError('simulated disk error')):
    try: release.publish(m,r,target)
    except OSError: pass
    else: raise AssertionError('Ожидалась ошибка атомарной замены')
assert target.read_bytes()==before,'Прежний указатель повреждён'
assert not list(root.glob('*.tmp')),'Остался временный файл'
'''
        result = subprocess.run([sys.executable, '-c', source, str(self.root)], cwd=self.tmp.name,
                                env={**os.environ, 'PYTHONPATH': str(PROJECT)}, text=True, capture_output=True)
        self.record('atomic-write-injection', returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        self.record('atomic-file-after', pointer_sha256=hashlib.sha256((self.root/'accepted.json').read_bytes()).hexdigest(),
                    temporary_files=[p.name for p in self.root.glob('*.tmp')])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_http_readiness_waits_for_warmup(self):
        self.server(seconds=2)
        self.assertEqual(self.call('/health')[0], 200)
        self.assertFalse(self.call('/debug')[1]['warmup_finished'])
        self.assertEqual(self.call('/ready')[0], 503)
        calls = self.call('/debug')[1]['calls']
        self.assertEqual(self.call('/generate', {'prompt': 'покажи будильники', 'max_new_tokens': 32})[0], 503)
        self.assertEqual(self.call('/debug')[1]['calls'], calls)
        self.wait_ready()
        self.assertEqual(self.call('/generate', {'prompt': 'покажи будильники', 'max_new_tokens': 32})[0], 200)

    def test_http_identity_restart_and_rollback(self):
        proc = self.server()
        self.wait_ready()
        request = {'prompt': 'покажи будильники', 'max_new_tokens': 32}
        self.assertEqual(self.call('/generate', request)[1]['model_id'], 'fixture-A')
        result = self.lab('run')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        old = self.call('/generate', request)[1]
        self.assertEqual(old['model_id'], 'fixture-A', 'Указатель B не загрузил веса B в память')
        self.assertIn('fixture-A', old['answer'])
        proc.terminate(); proc.wait(timeout=10)
        proc = self.server(); self.wait_ready()
        self.assertEqual(self.call('/generate', request)[1]['model_id'], 'fixture-B')
        result = self.lab('rollback', '--manifest', 'manifest-A.json', '--report', 'report-A.json')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call('/generate', request)[1]['model_id'], 'fixture-B')
        proc.terminate(); proc.wait(timeout=10)
        self.server(); self.wait_ready()
        self.assertEqual(self.call('/generate', request)[1]['model_id'], 'fixture-A')

    def test_http_validation_does_not_call_backend(self):
        self.server(); self.wait_ready()
        calls = self.call('/debug')[1]['calls']
        bad_requests = [
            {'prompt': '', 'max_new_tokens': 32}, {'prompt': '   ', 'max_new_tokens': 32},
            {'prompt': 'запрос', 'max_new_tokens': 0}, {'prompt': 'запрос', 'max_new_tokens': 100000},
            {'prompt': 'запрос', 'max_new_tokens': '32'}, {'prompt': 'запрос', 'max_new_tokens': True},
        ]
        for request in bad_requests:
            status, _ = self.call('/generate', request)
            self.assertTrue(400 <= status < 500, (request, status))
            self.assertEqual(self.call('/debug')[1]['calls'], calls)
        status, body = self.call('/generate', {'prompt': 'покажи будильники', 'max_new_tokens': 32})
        self.assertEqual(status, 200)
        self.assertIn('STUB ONLY', body['answer'])
        self.assertEqual(self.call('/debug')[1]['calls'], calls + 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
