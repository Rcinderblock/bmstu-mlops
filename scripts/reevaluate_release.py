"""При изменении условий заново оценить закреплённые веса, без нового sweep."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation import cli
from evaluation.contract import read_json
from src.release_bridge import historical_code

spec=read_json('metrics/candidate-model.json')
_,client=cli.tracking()
run=client.get_run(spec['run_id'])
if run.info.status!='FINISHED':raise ValueError('Обучение не завершено')
historical_code(run.data.tags)
manifest,_=cli.manifest_for(spec)  # закреплённые веса без нового выбора
import os
from src.runtime import resolve_device
if str(resolve_device(os.environ.get('HW6_DEVICE',manifest['model']['device'])))!=run.data.tags['device'] or os.environ.get('HW6_DTYPE',manifest['model']['dtype'])!=manifest['model']['dtype']:
    raise ValueError('Этот воспроизводитель закреплён на прежних device/dtype. Для миграции нужен отдельный runtime и полная связанная оценка, старый допуск не переносится.')
# Выбор модели уже зафиксирован. Новый код API не требует переобучать эти же веса.
cli.candidate_model=lambda:spec
cli.evaluate()
