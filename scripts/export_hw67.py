"""Собрать отчёты из реальных файлов оценки и записей сервера MLflow."""
from pathlib import Path
import argparse,csv,hashlib,json,random,shutil,sys
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mlflow import MlflowClient
import mlflow
from src.common import connect,load_params

ROOT=Path(__file__).resolve().parents[1]

def read(name):return json.loads((ROOT/name).read_text())
def dump(path,value):
 p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def pct(value):return f'{value:.1%}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]+['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows])

def main(log=False):
 p=load_params();connect(p);client=MlflowClient();sweep=read('metrics/sweep.json')
 a=read('metrics/baseline-eval.json');b=read('metrics/candidate-eval.json');gate=read('metrics/quality-gate.json');negative=read('metrics/negative-gate.json');storage=read('metrics/storage-check.json');diff=read('metrics/eval-diff.json');proof=read('data/eval/source-and-leakage.json')
 spec=read('metrics/candidate-model.json');aliases=dict(client.get_registered_model(spec['name']).aliases)
 runs=[client.get_run(row['run_id']) for row in sweep['rows']]
 wanted=['git_sha','git_dirty','code_sha256','config_sha256','data_train_sha256','data_val_sha256','base_revision','device','memory_metric','train_rows','val_rows','candidate','kind','mlflow.parentRunId']
 snapshots=[dict(run_id=r.info.run_id,status=r.info.status,parameters=r.data.params,metrics=r.data.metrics,tags={k:r.data.tags.get(k) for k in wanted},curves={key:[dict(step=m.step,value=m.value) for m in client.get_metric_history(r.info.run_id,key)] for key in ['train_loss','val_loss']}) for r in runs]
 dump('metrics/run-manifest.json',dict(parent_id=sweep['parent_id'],runs=snapshots,aliases=aliases))
 train=[json.loads(x) for x in (ROOT/'data/train.jsonl').read_text().splitlines()];val=[json.loads(x) for x in (ROOT/'data/val.jsonl').read_text().splitlines()]
 order=list(range(p['train']['train_limit']));random.Random(p['train']['seed']).shuffle(order)
 consumed=p['train']['max_steps']*p['train']['batch_size']*p['train']['grad_accum']
 budget=dict(train_pool=p['train']['train_limit'],val_rows=p['train']['val_limit'],steps=p['train']['max_steps'],consumed_rows=consumed,
  pool_labels=dict(Counter(r['messages'][-1]['content'] for r in train[:p['train']['train_limit']])),
  consumed_labels=dict(Counter(train[i]['messages'][-1]['content'] for i in order[:consumed])),
  val_labels=dict(Counter(r['messages'][-1]['content'] for r in val[:p['train']['val_limit']])),consumed_ids=[train[i]['id'] for i in order[:consumed]])
 dump('metrics/training-budget.json',budget)
 out=ROOT/'docs/results/hw67';out.mkdir(parents=True,exist_ok=True)
 names=['sweep','candidate-model','baseline-outputs','candidate-outputs','baseline-eval','candidate-eval','eval-diff','quality-gate','negative-gate','storage-check','release-decision','run-manifest','training-budget']
 for name in names:shutil.copyfile(ROOT/f'metrics/{name}.json',out/f'{name}.json')
 shutil.copyfile(ROOT/'data/eval/source-and-leakage.json',out/'data-version.json')
 for source,target in [('check-eval.log','hw67-tests.txt'),('legacy-tests.log','hw67-legacy-tests.txt')]:
  path=ROOT/'notes/work/hw06-07'/source
  if path.exists():shutil.copyfile(path,ROOT/'docs/results'/target)
 checked=(ROOT/'docs/results/hw67-check.txt').exists() and 'Итог: 9/9' in (ROOT/'docs/results/hw67-check.txt').read_text()
 status='Все девять исходных проверок прошли; отпечаток c42ac7788026.' if checked else 'Исходная приёмка будет выполнена после сборки отчёта.'
 best=next(r for r in runs if r.info.run_id==spec['run_id']);ratio=best.data.metrics['final_val_loss']/best.data.metrics['base_val_loss']
 metrics=table(['Измерение','База','Кандидат'],[["Правильные ответы",f"{a['overall']['correct']}/54",f"{b['overall']['correct']}/54"],["Accuracy",pct(a['overall']['accuracy']),pct(b['overall']['accuracy'])],["Допустимые JSON и метки",f"{a['overall']['valid']}/54",f"{b['overall']['valid']}/54"],["Format rate",pct(a['overall']['format_rate']),pct(b['overall']['format_rate'])],["Ошибки генерации",a['overall']['errors'],b['overall']['errors']],["Поведенческие случаи",f"{a['behavioral']['overall']['correct']}/5",f"{b['behavioral']['overall']['correct']}/5"]])
 slices=table(['Критический срез','n','Верно: база','Верно: кандидат','Accuracy: база → кандидат'],[[name,a['slices'][name]['n'],a['slices'][name]['correct'],b['slices'][name]['correct'],pct(a['slices'][name]['accuracy'])+' → '+pct(b['slices'][name]['accuracy'])] for name in ['alarm','calendar','lists','destructive']])
 rows={r['id']:r for r in map(json.loads,(ROOT/'data/eval/golden.jsonl').read_text().splitlines())};aa={r['id']:r for r in a['per_item']};bb={r['id']:r for r in b['per_item']}
 case_notes=[('golden-alarm_set-02','База добавила описание внутрь метки: JSON синтаксически корректен, но answer не принадлежит choices. Кандидат исправил именно этот дефект.'),('golden-calendar_remove-01','Удаляется календарная встреча. База выбрала calendar_remove; кандидат перепутал календарь с будильником. Общие слова remove не означают одинаковое действие.'),('golden-lists_remove-03','Удаляется пункт списка. Кандидат снова выбирает alarm_remove. Похожая формулировка удаления не делает метки взаимозаменяемыми.'),('golden-lists_query-06','Тема пункта — библиография, но запрос остаётся просмотром списка. REFUSE ошибочен: область задачи определяется действием, а не содержимым пункта.'),('golden-alarm_query-01','Обе модели путают просмотр существующего будильника с созданием. Улучшения здесь нет; такой общий промах тоже сохраняется в отчёте.')]
 pair_table=table(['Команда','Ожидается','База: raw','Кандидат: raw'],[[rows[i]['input'],rows[i]['expected'],aa[i]['raw'],bb[i]['raw']] for i,_ in case_notes])
 pairs='\n\n'.join(f"### {i}\n\nКоманда: {rows[i]['input']}.\n\nБаза: `{aa[i]['raw']}`. Кандидат: `{bb[i]['raw']}`.\n\n{note}" for i,note in case_notes)
 decision='Выпуск отклонён' if not gate['passed'] else 'Выпуск разрешён'
 conclusion=f'''# ДЗ6+7 — {decision.lower()}

## Выбор по ошибке

Один итоговый sweep `{sweep['parent_id']}` содержит шесть отдельных законченных запусков. Код обучения зафиксирован коммитом `{best.data.tags['git_sha']}`. Одинаковые seed 42, train-пул 32, val 8, четыре обновления, батч 1 × накопление 2; по факту используются восемь перемешанных примеров из пула. Test в выборе не участвует.

Лучший по loss — `{best.data.tags['candidate']}`, run `{spec['run_id']}`: r=4, lr=0.0003, q_proj/v_proj. Ошибка {best.data.metrics['base_val_loss']:.6f} → {best.data.metrics['final_val_loss']:.6f}, отношение {ratio:.6f} при допустимом максимуме 1.0. Версия `{spec['uri']}` получила alias `candidate`. Папка с токенизатором: {best.data.metrics['adapter_mb']:.2f} МиБ; сами матрицы меньше этой папки.

[Таблица шести запусков](comparison.csv), [их параметры, кривые и версии](results/hw67/run-manifest.json). Время `seconds` включает начальную и последующие оценки, обновления и отправку метрик; загрузка модели, сохранение и упаковка не входят. Память — выборочный максимум driver_allocated_memory, включая кэш драйвера. Запуски выполнялись последовательно в общем процессе: состояние аллокатора и загрузки проверочного пакета не изолированы. По этим единичным коротким измерениям нельзя причинно связывать разницу памяти или скорости только с lr/r.

## Данные и протокол до оценки

[Протокол](evaluation-protocol.md) фиксирует правило до получения ответов. Golden: 54 оригинальных команды, 19 связанных сценариев, по шесть примеров каждого из девяти намерений. Пять поведенческих строк считаются отдельно; пара перефразов имеет общую группу. Источник и поштучные пояснения меток сохранены в DVC; разбор выполнен с помощью агента, человеческая проверка не заявлена.

Полный индекс 2743 исходных вопросов покрывает train.pt в том же порядке id. DVC связывает исходные JSONL, код токенизации, точную ревизию и тензоры. Точных текстовых и групповых пересечений не найдено. Максимальный Jaccard по тройкам слов с train: {proof['max_train_golden_jaccard_3gram']:.2f}. Это не доказательство отсутствия всех смысловых перефразов. Хеши входов: [версия данных](results/hw67/data-version.json). Набор повторно используем для разработки; результат не является независимым внешним аудитом.

База — Qwen/Qwen3-0.6B, ревизия `{p['model']['revision']}`. Кандидат — `{spec['uri']}`. Одна system-инструкция, seed 42, greedy-декодирование, максимум 32 новых токена, рассуждения отключены. Expected не передаётся модели. Полный протокол SHA-256: `{a['protocol_sha256']}`. Реальная оценка двух версий сохранена в run `{b['evaluation_run_id']}`. Тестовые фикстуры не использовались вместо генерации.

## Общие метрики и срезы

{metrics}

{ slices }

Первые три среза не пересекаются; destructive пересекается с ними. Их n нельзя суммировать. Все неправильные, пустые и аварийные ответы входят в знаменатель; на этом прогоне ошибок генерации не было.

Исправлен {len(diff['improved'])} ответ, потеряно {len(diff['regressed'])}. Полные id: [diff](results/hw67/eval-diff.json). Сырые ответы: [база](results/hw67/baseline-outputs.json), [кандидат](results/hw67/candidate-outputs.json); per-item и counts: [база](results/hw67/baseline-eval.json), [кандидат](results/hw67/candidate-eval.json).

## Пять пар ответов

{pairs}

## Поведение, отказ и следующий шаг

База прошла 5/5 поведенческих случаев, кандидат — 4/5. Кандидат вернул корректный JSON, но перепутал добавление кабеля в список с установкой будильника. Оба перефраза просмотра списка и оба отказа на запрос вне темы прошли. Это отдельные обязательные случаи, а не ещё пять независимых golden-вопросов.

Правило отказало: `{gate.get('reason','PASS')}`. Дополнительно ухудшились calendar, lists и destructive и не пройден обязательный поведенческий случай. Улучшение format rate не компенсирует эти нарушения. [Отчёт гейта](results/hw67/quality-gate.json). Отрицательная проверка испорченной копии ответов тоже отклонена: `{negative['reason']}`; исходные ответы, веса и alias не менялись.

Итог: alias candidate указывает на версию {spec['version']}; champion отсутствует. Команда promote не выполнялась, baseline не подменялся. [Решение и состояние alias](results/hw67/release-decision.json).

Короткий пул не представляет всю задачу: в train-пуле только будильники {json.dumps(budget['pool_labels'],ensure_ascii=False)}, а в val: {json.dumps(budget['val_labels'],ensure_ascii=False)}. Наблюдаемое смещение ответов кандидата к alarm-меткам согласуется с этим ограничением; отдельно причинность не измеряли. Следующий шаг — собрать представительный стратифицированный train/val и одинаково увеличить бюджет всех конфигураций, затем заново проверить качество. Менять golden или порог ради принятия текущего кандидата нельзя.

## Ограничения сходства и модельного судьи

BLEU/ROUGE оценивают пересечение слов и фрагментов: alarm_remove и lists_remove похожи по написанию, но меняют разные сущности. BERTScore добавляет сходство представлений, однако тоже не гарантирует точного выполнения действия и корректного JSON. Для этой задачи нужен строгий scorer меток.

Модель-судья могла бы помочь разбирать неоднозначные формулировки и длинные свободные ответы. Ей нужна рубрика, скрытые названия моделей, перестановка A/B и B/A, сохранённые вердикты и проверка спорных случаев человеком. Возможны позиционное смещение, предпочтение многословия и ошибочные оценки. В этой работе платный API и модель-судья не применялись.

## Сохранность и проверки

Probe `{storage['run_id']}` пережил реальное пересоздание контейнера MLflow. SHA-256 `{storage['sha256']}` совпал. PostgreSQL и файлы артефактов используют разные постоянные тома. {status} Отдельно прошли 32 теста оценки и 40 прежних модульных тестов. Фикстуры проверяют код; числа выше получены от двух реальных моделей.

Ссылки на MLflow действуют при запущенном локальном сервере: [sweep](http://127.0.0.1:5006/#/experiments/2/runs/{sweep['parent_id']}), [оценка](http://127.0.0.1:5006/#/experiments/3/runs/{b['evaluation_run_id']}). Для преподавателя данные опубликованы отдельным приложением, без весов и тензоров обучения; инструкция — в README.
'''
 (ROOT/'docs/conclusion.md').write_text(conclusion)
 overview=f'''# ДЗ6+7 — эксперименты и качество ответов

## Задание

Исправить четыре дефекта, сравнить шесть конфигураций, сохранить переносимый пакет и отдельно проверить ответы базы и одного кандидата. Результат: {decision.lower()}. Candidate сохранён, champion отсутствует.

## Эксперименты

Один родительский запуск и шесть отдельных завершённых дочерних запусков. Четыре обновления, пул train 32, val 8, батч 1 × накопление 2, seed 42. Меняются lr, r и проекции. Полный индекс train содержит 2743 вопроса. Test не выбирает кандидата.

## Кандидат

| Поле | Значение |
|---|---|
| Конфигурация | qv-r4-high: r=4, lr=0.0003, q_proj/v_proj |
| Ошибка до → после | {best.data.metrics['base_val_loss']:.6f} → {best.data.metrics['final_val_loss']:.6f} |
| Отношение final/base, порог | {ratio:.4f}, максимум 1.0 |
| Реестр | {spec['uri']} |
| Alias | candidate → 1; champion отсутствует |

## Данные

54 golden-вопроса, 19 связанных сценариев. 2743 исходных вопроса train в полном индексе; все id связаны с train.pt. 0 точных и групповых пересечений. Максимальное сходство тройками слов: 0.25. Есть поштучные пояснения разметки с помощью агента; человеческая проверка не заявлена. Отсутствие всех смысловых перефразов не доказано.

## Протокол

Одинаковые system-инструкция и scorer. Seed 42, максимум 32 новых токена, do_sample=False, enable_thinking=False. Expected не включается в запрос. Строгий JSON с единственным answer из допустимых меток. Каждый вопрос остаётся в знаменателе. Поведенческие случаи считаются отдельно.

## Качество

{metrics}

{slices}

Пересекающиеся срезы не складываются в общий n. Исправлен 1 ответ, потеряно 7. Лучше формат не означает лучше смысл.

## Примеры

{pair_table}

## Решение

Гейт: отказ, {gate.get('reason','PASS')}. Порог не менялся. Кандидат остаётся доступен как candidate; champion не назначен, baseline не обновлён. Испорченная копия ответов также отклонена без изменения весов и alias. В небольшом обучающем пуле были только команды будильников: это важное ограничение опыта.

## Хранение

| Что | Где |
|---|---|
| Параметры, метрики и версии | Постоянный том PostgreSQL |
| Пакеты моделей и файлы | artifacts:/mlartifacts |
| Probe | {storage['run_id']} |
| SHA-256 до и после | {storage['sha256']} |

Сервис MLflow действительно пересоздан. Файл восстановлен с сервера без изменений. Приёмка также загружает candidate в отдельном процессе из временной папки, без локального конфига обучения.

## Проверки

{status} Дополнительно: 32 теста оценки и 40 прежних модульных тестов. Отказ по качеству — правильное решение о выпуске, не сбой технической приёмки. Тестовые фикстуры не заменяют оценку Qwen.

## Итог

| Файл | Результат |
|---|---|
| docs/comparison.csv | Шесть законченных экспериментов |
| docs/defects.md | Четыре причины и исправления |
| docs/conclusion.md | Один вывод по экспериментам и качеству |
| docs/results/hw67/ | Raw, per-item, diff, gate, negative, storage |
| evaluation/params.json | Правило оценки, закреплённое до измерения |
| evaluation/baseline.json | Исходная база с точной ревизией |

Репозиторий: https://github.com/Rcinderblock/bmstu-mlops. Данные проверки доступны отдельным приложением. Веса и тензоры, лекции, голос и записи экрана в Git не включены. На этом единая работа завершена.
'''
 (ROOT/'docs/overview.md').write_text(overview)
 defects=ROOT/'docs/defects.md';text=defects.read_text().split('\n## Подтверждения реальных запусков')[0]
 text=text.replace('Полные результаты и ссылка на проверку будут\nдобавлены после измерений.','Все шесть конфигураций завершились отдельно; таблица и версии записаны в отчётах ниже.').replace('результат фиксируется после запуска.','файл с тем же SHA-256 восстановлен после пересоздания.')
 text+=f'''\n## Подтверждения реальных запусков\n\n- Sweep: `{sweep['parent_id']}`; шесть дочерних runs FINISHED. Конфигурации r/lr/target_modules и кривые считываются с сервера, не вписаны вручную: [comparison.csv](comparison.csv), [run-manifest](results/hw67/run-manifest.json).\n- Версии входов: train `{best.data.tags['data_train_sha256']}`, val `{best.data.tags['data_val_sha256']}`. База `{best.data.tags['base_revision']}`, код `{best.data.tags['git_sha']}`.\n- Probe после замены контейнера: `{storage['run_id']}`. SHA-256 `{storage['sha256']}` совпал; [storage-check](results/hw67/storage-check.json).\n- Лучший final/base = {ratio:.6f} при пороге 1.0. Проверка порога 0.0 обязана отклонять регистрацию до вызова API записи. {status}\n- Дальнейший гейт качества тоже дал отказ: {gate.get('reason','PASS')}. Это отдельный этап: candidate не равен champion.\n'''
 defects.write_text(text)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(13,4.6))
 for snap in snapshots:
  points=snap['curves']['val_loss'];axes[0].plot([x['step'] for x in points],[x['value'] for x in points],marker='o',label=snap['tags']['candidate'])
 axes[0].set(xlabel='Обновление весов',ylabel='Ошибка на 8 отложенных примерах',title='Реальные кривые шести конфигураций');axes[0].legend(fontsize=8,ncol=2);axes[0].grid(alpha=.2)
 ranked=sorted(snapshots,key=lambda x:x['metrics']['final_val_loss']);axes[1].barh([s['tags']['candidate'] for s in ranked],[s['metrics']['final_val_loss'] for s in ranked],color=['#367d69' if s['run_id']==spec['run_id'] else '#597fb0' for s in ranked]);axes[1].axvline(best.data.metrics['base_val_loss'],color='#b94b4b',linestyle='--',label='База до обучения');axes[1].invert_yaxis();axes[1].set(xlabel='Итоговая ошибка: меньше — лучше',title='Выбор кандидата по одному mini-val');axes[1].legend(fontsize=9);fig.tight_layout();fig.savefig(ROOT/'docs/experiment-curves.png',dpi=160);plt.close(fig)
 readme=f'''# Русские команды личного планировщика — ДЗ6+7

**Работа выполнена; выпуск кандидата отклонён по качеству.** Шесть отдельных коротких экспериментов сохранены в MLflow. Лучший по loss адаптер получил alias `candidate`, но ухудшил ответы и не назначен `champion`. Пороги после результата не менялись.

{metrics}

Исправлен один ответ, потеряно семь. Календарь: 6/18 → 2/18; удаление: 10/18 → 6/18. Все вопросы остались в знаменателе. [Полный вывод с пятью парами ответов](docs/conclusion.md).

## Результаты для сдачи

- [Шесть экспериментов](docs/comparison.csv), [их кривые](docs/experiment-curves.png).
- [Четыре дефекта](docs/defects.md), [единый вывод](docs/conclusion.md), [протокол до оценки](docs/evaluation-protocol.md).
- [Raw, per-item, diff, gate, negative, storage и версии](docs/results/hw67/).
- [Данные проверки отдельным приложением](https://github.com/Rcinderblock/bmstu-mlops/releases/download/hw67-eval-v1/hw67-evaluation-data-v1.zip): 54 golden, 5 behavioral, полный индекс 2743 train-вопросов, источник и разбор меток. SHA-256 закреплён в `conf/evaluation-data.json`; `make fetch-eval` проверяет архив до распаковки. Весов и тензоров в нём нет.

{status} 32 проверки оценки и 40 прежних модульных тестов прошли. Искусственные ответы тестов не используются вместо реальной оценки моделей. Исходные файлы `tests/` сохранены без изменений, отпечаток **c42ac7788026**. Дополнительные проверки — в `tests/project/`, прежние — в `tests/legacy/`.

## Повторение

Нужны Python 3.12, uv, Git, Docker Compose v2 и память для Qwen3-0.6B. На первом запуске база скачивается с Hugging Face. После заполнения кэша можно задать `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

```sh
make install
make up                 # MLflow на 127.0.0.1:5006, два постоянных тома
make fetch-data
make data               # исходные вопросы → train.pt / val.pt
make fetch-eval         # опубликованный проверочный набор с SHA-256
make check-eval         # тесты кода оценки без генерации модели
make sweep              # 6 конфигураций × 4 обновления
make report
make candidate          # loss-порог → candidate
make evaluate           # реальные ответы базы и одного кандидата
make eval               # может штатно отказать при ухудшении
make check-negative     # испорченная копия ответов обязана получить отказ
make check-storage      # реально пересоздать только наш сервер MLflow
make check-all          # 9 исходных проверок и 32 модульные проверки оценки
```

`evaluation/baseline.json` уже сохранён в Git. `make baseline-init` нужен только при самом первом создании протокола и отказывается перезаписывать существующую базу. `make promote` допускается только после PASS качества; в текущем результате его выполнять нельзя. `make baseline-update` применяется явно после принятия версии, сохраняя историю.

`make down` сохраняет Docker-тома; `down -v` удаляет их и не используется для обычной остановки. MLflow и ссылки на runs локальные, не публичный облачный сервис. На другом ноутбуке результаты нужно восстановить из своего хранилища либо повторить команды. Веса не включены в Git и приложение.

## Файлы и ограничения

`src/sweep.py` создаёт отдельные runs; `src/experiment_train.py` читает тензоры, обучает добавочные матрицы и сохраняет пакет. `src/registry_model.py` загружает его без локального конфига. `evaluation/solution.py` считает строгие метки и запрещает регрессию. `scripts/prepare_evaluation.py`, `evaluation-source.dvc` и `dvc.yaml` связывают источник, полный train-индекс и проверочные данные.

Бюджет учебный: pool 32, val 8, 4 обновления по два примера. Пул содержит только команды будильников; это не полноценное обучение на всей теме. Память включает кэш драйвера, а время измерено по одному короткому запуску. Ошибка на mini-val не заменяет оценку ответов. Golden содержит связанные сценарии; отсутствие всех смысловых утечек не доказано.

Обучающий источник: [Amazon MASSIVE 1.0](https://github.com/alexa/massive), [CC BY 4.0](docs/MASSIVE-LICENSE.txt). Новые примеры составлены и разобраны с помощью агента; человеческая проверка не заявлена. Согласование темы преподавателем не зафиксировано.

Оригиналы лекций, заметки, окружение, локальные данные, веса, образец голоса и рабочие записи исключены из Git. Сохранённые этапы семестра доступны в истории коммитов; старые конфиги — в `conf/history/`. Исторические полные проверки выполняйте в копии соответствующего коммита, а модульные — `make test-legacy`.
'''
 (ROOT/'README.md').write_text(readme)
 if log:
  evaluation_run_id=b['evaluation_run_id']
  client.log_artifacts(evaluation_run_id,str(out),'evaluation/evidence')
  for name in ['docs/conclusion.md','docs/defects.md','docs/evaluation-protocol.md','docs/comparison.csv','docs/experiment-curves.png','conf/evaluation-data.json']:
   client.log_artifact(evaluation_run_id,str(ROOT/name),'evaluation/reports')
 print('Exported actual reports; gate:',gate,'aliases:',aliases,'technical acceptance:',checked)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--log',action='store_true');main(ap.parse_args().log)
