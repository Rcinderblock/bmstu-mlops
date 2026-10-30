"""Локальный FastAPI: фактически загруженная версия, готовность и один вычислитель."""
import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from src.api_backend import StubBackend, TransformersBackend
from src.deployment import load_accepted


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: StrictStr = Field(min_length=1, max_length=2000)
    max_new_tokens: StrictInt = Field(default=32, ge=1, le=32)

    @field_validator('prompt')
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError('Нужен непустой prompt')
        return value


class ServiceState:
    def __init__(self, root, warmup_seconds, backend_factory=None):
        self.root = Path(root).resolve()
        self.backend = None
        self.ready = False
        self.error = None
        self.warmup_finished = threading.Event()
        self.semaphore = threading.BoundedSemaphore(1)
        self.closed = threading.Event()

        def load():
            try:
                manifest = load_accepted(self.root)
                factory = backend_factory or {'stub': StubBackend, 'transformers': TransformersBackend}[manifest['backend']]
                if factory.kind != manifest['backend']:
                    raise ValueError('Backend не соответствует принятому manifest')
                self.backend = factory(manifest, self.root)
                if self.closed.wait(warmup_seconds):
                    return
                self.backend.warmup()
                self.warmup_finished.set()
                self.ready = True
            except Exception as exc:
                self.error = f'{type(exc).__name__}: {exc}'
                print('LOAD_FAILED', self.error, flush=True)
        self.loader = threading.Thread(target=load, daemon=True, name='model-loader')
        self.loader.start()

    def generate(self, request, request_id):
        if not self.ready or self.backend is None:
            raise HTTPException(503, 'Модель ещё не загружена и не прогрета')
        limit = self.backend.runtime['generation']['max_new_tokens']
        if request.max_new_tokens != limit:
            raise HTTPException(422, 'Разрешён только оценённый лимит: ' + str(limit))
        started = time.perf_counter()
        with self.semaphore:
            acquired = time.perf_counter()
            try:
                result = self.backend.generate(request.prompt, limit)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            result.update(model_id=self.backend.loaded_id, runtime_sha256=self.backend.runtime_sha256,
                          request_id=request_id, queue_seconds=acquired-started,
                          service_seconds=time.perf_counter()-started, backend=self.backend.kind)
            return result


def create_app(root, warmup_seconds=0, backend_factory=None, debug=False):
    state_holder = {}

    @asynccontextmanager
    async def lifespan(app):
        state = ServiceState(root, warmup_seconds, backend_factory)
        state_holder['state'] = state
        yield
        state.ready = False
        state.closed.set()
        state.loader.join(timeout=5)

    app = FastAPI(title='Личный планировщик — выпуск модели', lifespan=lifespan)

    @app.middleware('http')
    async def request_metadata(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        raw = request.headers.get('content-length')
        if raw is not None:
            try:
                size = int(raw)
                if size < 0 or size > 65536:
                    return JSONResponse({'error': 'body too large', 'request_id': request.state.request_id}, status_code=413)
            except ValueError:
                return JSONResponse({'error': 'invalid content-length'}, status_code=400)
        response = await call_next(request)
        response.headers['X-Request-ID'] = request.state.request_id
        return response

    @app.get('/health')
    def health():
        return {'alive': True}

    @app.get('/ready')
    def ready():
        state = state_holder['state']
        body = dict(ready=state.ready)
        if state.ready:
            body.update(model_id=state.backend.loaded_id, runtime_sha256=state.backend.runtime_sha256)
        if state.error:
            body['load_failed'] = True
        return JSONResponse(body, status_code=200 if state.ready else 503)

    @app.get('/debug', include_in_schema=False)
    def diagnostics():
        state = state_holder['state']
        backend = state.backend
        if not debug and (backend is None or backend.kind != 'stub'):
            raise HTTPException(404)
        return dict(loaded_id=backend.loaded_id if backend else None, calls=backend.calls if backend else 0,
                    warmup_finished=state.warmup_finished.is_set(), load_error=state.error)

    @app.post('/generate')
    def generate(payload: GenerationRequest, request: Request):
        return state_holder['state'].generate(payload, request.state.request_id)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('.lab'))
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--warmup-seconds', type=float, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.warmup_seconds < 0:
        parser.error('warmup-seconds должен быть неотрицательным')
    import uvicorn
    print('API http://127.0.0.1:' + str(args.port), flush=True)
    uvicorn.run(create_app(args.workspace, args.warmup_seconds, debug=args.debug),
                host='127.0.0.1', port=args.port, log_level='warning', access_log=False)


if __name__ == '__main__':
    main()
