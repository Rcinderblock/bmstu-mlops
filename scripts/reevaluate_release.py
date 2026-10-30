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
cli.manifest_for(spec)  # реальная загрузка именно закреплённой версии артефакта
# Выбор модели уже зафиксирован. Новый код API не требует переобучать эти же веса.
cli.candidate_model=lambda:spec
cli.evaluate()
