"""CPU-имитатор для Kubernetes: настоящие HTTP, очередь, CPU и потоковые части."""
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, StrictStr
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, ProcessCollector, generate_latest

BUCKETS=(.01,.025,.05,.1,.2,.4,.8,1.2,2,3,5,8,15,30,60)
class Message(BaseModel):
    role: StrictStr = Field(pattern='^(user|assistant|system)$')
    content: StrictStr = Field(min_length=1,max_length=2000)
class StreamRequest(BaseModel):
    request_id: str = Field(default_factory=lambda:str(uuid.uuid4()),max_length=100)
    session_id: str = Field(default='anonymous',max_length=100)
    messages: list[Message] = Field(min_length=1,max_length=16)
    parts: int = Field(default=4,ge=1,le=100,strict=True)
    chunk_ms: int = Field(default=25,ge=1,le=1000,strict=True)


def cpu_work(seconds):
    """Время именно CPU-потока; sleep не подменяет вычислительную нагрузку HPA."""
    end=time.thread_time()+seconds
    raw=b'cpu-simulated-model'*4096
    while time.thread_time()<end:raw=hashlib.sha256(raw).digest()*2048


def create_simulated_app():
    pod=os.getenv('POD_NAME','local')
    registry=CollectorRegistry()
    ProcessCollector(registry=registry)
    total=Counter('planner_requests','All admitted or rejected generation attempts',['outcome'],registry=registry)
    first=Histogram('planner_ttfc_seconds','Time from server receipt to first chunk',buckets=BUCKETS,registry=registry)
    duration=Histogram('planner_duration_seconds','Successful full stream duration',buckets=BUCKETS,registry=registry)
    queue=Histogram('planner_queue_seconds','Wait for compute slot',buckets=BUCKETS,registry=registry)
    running=Gauge('planner_running','Active streams',registry=registry)
    waiting=Gauge('planner_waiting','Queued requests',registry=registry)
    ready_metric=Gauge('planner_ready','Loaded and warmed',registry=registry)
    cache_events=Counter('planner_cache','Simulated local prefix cache',['result'],registry=registry)
    state={'ready':False,'draining':False,'error':None,'model_id':None,'artifact_sha256':None,'waiting':0,'running':0,'calls':0}
    cache=OrderedDict();executor=ThreadPoolExecutor(max_workers=int(os.getenv('MAX_RUNNING','1')))
    semaphore=asyncio.Semaphore(int(os.getenv('MAX_RUNNING','1')))
    def log(event,**fields):print(json.dumps(dict(event=event,wall_time=time.time(),pod=pod,backend_mode='simulated',model_id=state['model_id'],**fields),ensure_ascii=False),flush=True)
    async def load():
        try:
            raw=Path(os.getenv('MODEL_FILE','/model/model.json')).read_bytes()
            actual=hashlib.sha256(raw).hexdigest()
            if actual!=os.environ['MODEL_SHA256']:raise ValueError('artifact checksum mismatch')
            model=json.loads(raw);assert model['backend_mode']=='simulated'
            state.update(model_id=model['model_id'],artifact_sha256=actual)
            log('loaded',artifact_sha256=actual)
            await asyncio.sleep(float(os.getenv('WARMUP_SECONDS','20')))
            if os.getenv('NEVER_READY','false')=='true':raise ValueError('deliberately unready release')
            await asyncio.get_running_loop().run_in_executor(executor,cpu_work,.05)
            state['ready']=True;ready_metric.set(1);log('warmup_finished')
        except Exception as exc:state['error']=str(exc);log('load_failed',error=str(exc))
    @asynccontextmanager
    async def lifespan(app):
        task=asyncio.create_task(load());yield
        task.cancel();executor.shutdown(wait=False,cancel_futures=True)
    app=FastAPI(title='Planner Kubernetes CPU simulator',lifespan=lifespan)
    @app.middleware('http')
    async def identity_headers(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers.update({'X-Pod-Name':pod,'X-Model-ID':state['model_id'] or 'not-loaded','X-Backend-Mode':'simulated','X-Request-ID':request.state.request_id})
        return response
    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc):
        return JSONResponse(dict(error=exc.detail,backend_mode='simulated',pod=pod,model_id=state['model_id'],request_id=request.state.request_id),status_code=exc.status_code)
    @app.get('/health')
    def health():return {'alive':True,'pod':pod}
    @app.get('/ready')
    @app.get('/startup')
    def ready():
        ok=(state['ready'] and not state['draining']) or os.getenv('BROKEN_PROBES','false')=='true'
        return JSONResponse(dict(state,backend_mode='simulated',pod=pod),status_code=200 if ok else 503)
    @app.get('/debug')
    def debug():return dict(state,pod=pod,cache_entries=len(cache),backend_mode='simulated')
    @app.post('/admin/reset-cache')
    def reset():cache.clear();return {'cache_entries':0}
    @app.post('/admin/drain')
    async def drain():
        state['draining']=True;ready_metric.set(0);log('draining');await asyncio.sleep(3);return {'draining':True}
    @app.get('/metrics')
    def metrics():return Response(generate_latest(registry),media_type='text/plain; version=0.0.4')
    @app.post('/generate')
    async def generate(payload:StreamRequest,request:Request):
        received=time.perf_counter();rid=payload.request_id;request.state.request_id=rid
        identity=dict(request_id=rid,session_id=payload.session_id,client_ip=request.client.host)
        if not state['ready'] or state['draining']:
            total.labels('unready').inc();log('rejected_unready',**identity)
            raise HTTPException(503,'backend is not warmed or draining')
        if state['waiting']>=int(os.getenv('MAX_WAITING','32')):
            total.labels('rejected').inc();log('rejected_queue',**identity);raise HTTPException(429,'queue full')
        state['waiting']+=1;waiting.inc();log('queued',**identity)
        try:await asyncio.wait_for(semaphore.acquire(),timeout=float(os.getenv('QUEUE_TIMEOUT','8')))
        except TimeoutError:
            total.labels('rejected').inc();log('rejected_timeout',**identity);raise HTTPException(429,'queue timeout')
        finally:state['waiting']-=1;waiting.dec()
        q=time.perf_counter()-received;queue.observe(q);state['running']+=1;running.inc();state['calls']+=1
        history=[m.model_dump() for m in payload.messages]
        prefix=hashlib.sha256(json.dumps(history[:-1] or history,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        hit=prefix in cache;cache_events.labels('hit' if hit else 'miss').inc()
        cache[prefix]=True;cache.move_to_end(prefix)
        while len(cache)>int(os.getenv('CACHE_ENTRIES','8')):cache.popitem(last=False)
        async def stream():
            complete=False;n=0
            shared=dict(request_id=rid,pod=pod,model_id=state['model_id'],artifact_sha256=state['artifact_sha256'],backend_mode='simulated',queue_seconds=q,cache_hit=hit)
            try:
                cost=float(os.getenv('CACHE_HIT_CPU_MS','5') if hit else os.getenv('CACHE_MISS_CPU_MS','80'))/1000
                await asyncio.get_running_loop().run_in_executor(executor,cpu_work,cost)
                for n in range(payload.parts):
                    await asyncio.sleep(payload.chunk_ms/1000)
                    if n==0:first.observe(time.perf_counter()-received)
                    yield json.dumps(dict(shared,index=n,text=f'simulated part {n}',done=False))+'\n'
                response={'history_messages':len(history),'history_sha256':hashlib.sha256(json.dumps(history,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),'last_user':next((m['content'] for m in reversed(history) if m['role']=='user'),'')}
                complete=True;duration.observe(time.perf_counter()-received);total.labels('completed').inc()
                yield json.dumps(dict(shared,done=True,parts=payload.parts,response=response,service_seconds=time.perf_counter()-received),ensure_ascii=False)+'\n'
                log('completed',**identity,queue_seconds=q,elapsed=time.perf_counter()-received,parts=payload.parts,cache_hit=hit)
            finally:
                if not complete:total.labels('interrupted').inc();log('interrupted',**identity,parts_sent=n)
                state['running']-=1;running.dec();semaphore.release()
        return StreamingResponse(stream(),media_type='application/x-ndjson',headers={'X-Pod-Name':pod,'X-Model-ID':state['model_id'],'X-Backend-Mode':'simulated','X-Request-ID':rid})
    return app
