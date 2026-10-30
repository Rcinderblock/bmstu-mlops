"""Четыре задачи Prefect: подготовка → артефакт → допуск → публикация."""
from pathlib import Path
import uuid

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.context import get_run_context
from prefect.filesystems import LocalFileSystem

from src import deployment as core


def input_key(context, parameters):
    # Отдельные стенды не заимствуют путь результата из чужого удаляемого workspace.
    # Содержательная идентичность остаётся path-independent в release.cache_key.
    from release import canonical, digest
    root = str(Path(parameters['root']).resolve())
    return digest(canonical(dict(storage_scope=root, content_key=core.preparation_key(root))))


@task(name='validate-input', persist_result=True, cache_key_fn=input_key,
      result_serializer='json', cache_result_in_memory=False)
def prepare_input(root):
    return core.prepare(root)


def transient_only(task, task_run, state):
    try:
        state.result()
    except (ConnectionError, TimeoutError):
        return True
    except Exception:
        return False
    return False


@task(name='retrieve-artifact', cache_policy=NO_CACHE, retries=2,
      retry_delay_seconds=[.2, .5], retry_condition_fn=transient_only)
def retrieve_artifact(root, manifest_name):
    manifest = core.read(core.safe_path(root, manifest_name))
    verification = core.verify_artifact(root, manifest)
    return dict(manifest=manifest, verification=verification,
                operation='read-existing-artifact; no training')


@task(name='evaluate-release', cache_policy=NO_CACHE, retries=0)
def evaluate_release(root, artifact, report_name, prepared):
    return core.assess(root, artifact['manifest'], report_name, prepared)


@task(name='publish-accepted', cache_policy=NO_CACHE, retries=0)
def publish_accepted(root, artifact, decision, fail_after_pointer):
    return core.publish_once(root, artifact['manifest'], decision, fail_after_pointer)


@flow(name='planner-release', log_prints=False, retries=0)
def release_flow(root, manifest_name='manifest-B.json', report_name='report-B.json',
                 fail_after_prepare=False, fail_after_pointer=False):
    root = str(Path(root).resolve())
    run_id = str(get_run_context().flow_run.id)
    attempt = str(uuid.uuid4())
    core.append_attempt(root, 'started', attempt_id=attempt, flow_run_id=run_id)
    print('FLOW_RUN_ID', run_id, flush=True)

    def record(name, state):
        core.append_attempt(root, 'task', attempt_id=attempt, flow_run_id=run_id,
                            task=name, state=state.name, task_run_id=str(state.state_details.task_run_id))
        return state.result()

    try:
        core.recover(root)
        storage = LocalFileSystem(basepath=str(Path(root) / '.prefect-results'))
        from release import digest
        storage.save('planner-results-' + digest(root.encode())[:20], overwrite=True)
        state = prepare_input.with_options(result_storage=storage)(root, return_state=True)
        prepared = record('prepare', state)
        print('PREPARE_TASK_STATE', state.name, flush=True)
        if state.name == 'Cached':
            print('CACHE_HIT', prepared['input_sha256'], flush=True)
        if fail_after_prepare:
            raise RuntimeError('Инъекция сбоя после подготовки')
        artifact = record('artifact', retrieve_artifact(root, manifest_name, return_state=True))
        decision = record('gate', evaluate_release(root, artifact, report_name, prepared, return_state=True))
        result = record('publish', publish_accepted(root, artifact, decision, fail_after_pointer, return_state=True))
        core.append_attempt(root, 'completed', attempt_id=attempt, flow_run_id=run_id, **result)
        return dict(flow_run_id=run_id, prepare_state=state.name, **result)
    except Exception as exc:
        core.append_attempt(root, 'failed', attempt_id=attempt, flow_run_id=run_id,
                            error=f'{type(exc).__name__}: {exc}')
        raise
