"""Собрать отчёт только из сохранённых результатов настоящих опытов."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'notes/work/hw08-09-10'
OUT=ROOT/'docs/results/hw8910'
MODEL=ROOT/'deployments/planner'
FIXED='55e044d8400e8c43d9fa4c00bbd101d021faac3d'
ORIGINAL='0f1fce9'


def read(path):return json.loads(Path(path).read_text())
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]+['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    copies={'original-contract-tests.log':'tests-contract-before.txt','original-integration-tests-final.log':'tests-integration-before.txt',
            'final-fast-tests.log':'tests-fast-after.txt','intentional-regression.log':'tests-intentional-failure.txt',
            'previous-evaluation-tests.log':'tests-evaluation-regression.txt','previous-legacy-tests.log':'tests-legacy-regression.txt'}
    for source,target in copies.items():shutil.copyfile(WORK/source,OUT/target)
    for source,target in [('evidence-before-final','before'),('evidence-after-final','after'),('experiments','flow-experiments'),('api','api')]:
        dest=OUT/target;dest.mkdir(exist_ok=True)
        for p in (WORK/source).glob('*'):
            if p.is_file():shutil.copyfile(p,dest/p.name)
    for name in ['manifest-baseline.json','manifest-candidate.json','release-report-baseline.json','release-report-candidate.json','accepted.json','attempts.jsonl','publications.jsonl','tokenizer-equivalence.json']:
        shutil.copyfile(MODEL/name,OUT/name)
    shutil.copyfile(MODEL/'evidence/origin.json',OUT/'origin.json')
    for p in OUT.rglob('*.log'):
        p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    for p in OUT.glob('*.txt'):
        p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    diff=subprocess.check_output(['git','diff',ORIGINAL,FIXED,'--','release.py','lab.py','service.py','src/deployment.py','src/release_flow.py','src/api_backend.py','src/release_bridge.py','src/prefect_local.py'],cwd=ROOT,text=True)
    (OUT/'fixes.patch').write_text(diff)
    assert 'Ran 20 tests' in (OUT/'tests-fast-after.txt').read_text() and '\nOK\n' in (OUT/'tests-fast-after.txt').read_text()
    assert 'Ran 7 tests' in (OUT/'tests-integration-before.txt').read_text() and 'FAILED' in (OUT/'tests-integration-before.txt').read_text()
    expected=['test_atomic_failure_preserves_real_file_bytes','test_cache_recovery_and_changed_bytes_in_new_process','test_http_identity_restart_and_rollback','test_http_readiness_waits_for_warmup','test_http_validation_does_not_call_backend','test_identical_publication_and_repair_after_pointer_crash','test_incomplete_and_foreign_reports_preserve_pointer']
    for name in expected:
        assert ('FAIL: '+name) in (OUT/'tests-integration-before.txt').read_text(),name
    budget=read(OUT/'manifest-baseline.json');config=read(MODEL/'artifacts/base/config.json')
    kv_bytes=2*config['num_hidden_layers']*config['num_key_value_heads']*config.get('head_dim',config['hidden_size']//config['num_attention_heads'])*2
    hardware=dict(python=platform.python_version(),system=platform.platform(),machine=platform.machine(),
                  cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                  memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),
                  device=budget['runtime']['device'],dtype=budget['runtime']['dtype'],
                  theoretical_kv_bytes_per_token_per_sequence=kv_bytes,kv_is_estimate_not_measurement=True)
    dump(OUT/'hardware.json',hardware)
    base=read(OUT/'release-report-baseline.json');candidate=read(OUT/'release-report-candidate.json')
    assert base['passed'] is True and candidate['passed'] is False
    experiments=read(OUT/'flow-experiments/results.json')
    assert all(r['pointer_sha256_before']==r['pointer_sha256_after'] for r in experiments if r['exit_code']!=0)
    events=[json.loads(x) for x in (MODEL/'publications.jsonl').read_text().splitlines()]
    assert len(events)==1
    with sqlite3.connect(MODEL/'publications.sqlite3') as db:
        journal=[dict(release_id=r[0],model_id=r[1],state=r[2]) for r in db.execute('SELECT release_id,model_id,state FROM releases')]
    dump(OUT/'journal-state.json',journal)
    batches=read(OUT/'api/summary.json');attempts=[json.loads(x) for x in (OUT/'api/attempts.jsonl').read_text().splitlines()]
    assert len(attempts)==16 and all(b['successes']==8 and b['errors']==0 for b in batches)
    a={r['case_id']:r for r in attempts if r['concurrency']==1};b={r['case_id']:r for r in attempts if r['concurrency']==2}
    assert set(a)==set(b) and all(a[k]['response']['answer']==b[k]['response']['answer'] for k in a)
    fields=['attempt_id','case_id','concurrency','max_new_tokens','status','error','client_seconds','request_id','model_id','runtime_sha256','generation_seconds','queue_seconds','output_tokens','answer']
    with (ROOT/'docs/load-results.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in attempts:
            body=r['response'] or {};row={k:r.get(k) for k in fields}
            row.update({k:body.get(k) for k in fields if k in body});w.writerow(row)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    plt.rcParams.update({'font.size':12})
    labels=['1 одновременно','2 одновременно']
    axes[0].bar(labels,[r['mean_success_latency'] for r in batches],color=['#527ba0','#c58a38']);axes[0].set_title('Средняя задержка клиента');axes[0].set_ylabel('секунды')
    for i,r in enumerate(batches):axes[0].text(i,r['mean_success_latency']+.015,f"{r['mean_success_latency']:.3f}",ha='center')
    axes[1].bar(labels,[r['mean_generation_seconds'] for r in batches],label='вычисление',color='#527ba0')
    axes[1].bar(labels,[r['mean_queue_seconds'] for r in batches],bottom=[r['mean_generation_seconds'] for r in batches],label='ожидание',color='#c58a38');axes[1].set_title('Время внутри сервиса');axes[1].set_ylabel('секунды');axes[1].legend()
    axes[2].bar(labels,[r['successes_per_second'] for r in batches],color='#507d67');axes[2].set_title('Пропускная способность');axes[2].set_ylabel('HTTP 200 / с')
    for i,r in enumerate(batches):axes[2].text(i,r['successes_per_second']+.03,f"{r['successes_per_second']:.2f}",ha='center')
    fig.suptitle('Одинаковые 8 запросов × 2 режима; один вычислитель; HTTP 200 не означает правильную метку')
    fig.savefig(ROOT/'docs/load-results.png',dpi=140);plt.close(fig)
    rows=[(1,'Старый кеш после изменения файла','run; изменить input.txt; run','Новая подготовка','CACHE_HIT старых байтов','Ключ хешировал только путь','test_cache_recovery_and_changed_bytes_in_new_process'),
          (2,'Неполный/чужой отчёт разрешал выпуск','run --report report-incomplete.json','Отказ и прежний SHA указателя','PUBLISHED fixture-B','Проверка истинности вместо полного контракта и связанного допуска','test_incomplete_and_foreign_reports_preserve_pointer'),
          (3,'Ошибка замены не сохраняла прежний файл','test_atomic_failure_preserves_real_file_bytes','OSError; прежние байты; нет *.tmp','Записан B без os.replace','Прямая перезапись accepted.json','test_atomic_failure_preserves_real_file_bytes'),
          (4,'Повтор размножал выпуск','run; run','Одно событие B','Две строки публикаций','Безусловный append без устойчивого release_id','test_identical_publication_and_repair_after_pointer_crash'),
          (5,'Готовность до прогрева','service.py --warmup-seconds 2; GET /ready','503 до прогрева','200 при warmup_finished=false','ready=True до загрузки/прогрева','test_http_readiness_waits_for_warmup'),
          (6,'Ответ рекламировал B при загруженной A','запустить A; run B; POST /generate','model_id=A до перезапуска','В тексте A но model_id=B','Чтение указателя при каждом ответе','test_http_identity_restart_and_rollback'),
          (7,'Неверный запрос запускал backend','POST /generate с пустым prompt/неверным лимитом','4xx и прежний calls','200 и новый вызов','Нет строгой валидации до вычисления','test_http_validation_does_not_call_backend')]
    fixes=['SHA-256 входа и всех параметров плюс версия обработки; постоянный результат',
           'Строгий контракт и повторный вызов прежнего quality_gate по проверенным полным отчётам',
           'Временный файл рядом, flush/fsync, os.replace, уборка при ошибке',
           'SQLite: уникальный release_id и pending/committed; восстановление JSONL-проекции',
           'Фоновая загрузка и настоящий прогрев; независимые health/ready',
           'ID/runtime из загруженного backend; управляемый перезапуск и проверенный откат',
           'Строгие типы, длина, фиксированный оценённый лимит до вызова backend']
    with (ROOT/'defects.csv').open('w',newline='') as stream:
        w=csv.writer(stream,lineterminator='\n');w.writerow(['id','symptom','reproduce','expected','observed','root_cause','fix_commit','regression_test','before_log','after_log','fix'])
        for (idx,sym,command,exp,obs,cause,test),fix in zip(rows,fixes):
            w.writerow([idx,sym,command,exp,obs,cause,FIXED,'IncidentIntegrationTests.'+test,'docs/results/hw8910/before/'+test+'.jsonl','docs/results/hw8910/after/'+test+'.jsonl',fix])
    defect_table=table(['№','Причина','Исправление'],[[r[0],r[5],f] for r,f in zip(rows,fixes)])
    test_table=table(['Проверка','До','После'],[[r[6],'FAIL','ok'] for r in rows])
    batch_table=table(['Параллельность','n','HTTP 200','Ошибок','Весь пакет, с','HTTP 200/с','Средняя, с','Максимальная, с','Очередь, с','Вычисление, с','Выходных токенов'],[[r['concurrency'],r['n'],r['successes'],r['errors'],f"{r['batch_seconds']:.3f}",f"{r['successes_per_second']:.3f}",f"{r['mean_success_latency']:.3f}",f"{r['max_success_latency']:.3f}",f"{r['mean_queue_seconds']:.3f}",f"{r['mean_generation_seconds']:.3f}",r['output_tokens']] for r in batches])
    flow_table=table(['Опыт','Exit','Подготовка','Прежний указатель при отказе'],[[r['name'],r['exit_code'],'CACHE_HIT' if r['cache_hit'] else 'PREPARED' if r['prepared'] else '—','да' if r['exit_code'] and r['pointer_sha256_before']==r['pointer_sha256_after'] else '—'] for r in experiments])
    lifecycle=read(OUT/'api/lifecycle.json');smoke=next(r for r in lifecycle if r['event']=='real-smoke')['response']
    readiness=table(['Событие','HTTP','Фактическая версия'],[[r['event'],r['status'],r['response'].get('model_id','—')] for r in lifecycle if r['event'] in ['initial-health','initial-not-ready','initial-ready','restart-ready','restart-real-smoke']])
    methods=table(['Событие','Действие и команды'],[
        ['Код HTTP/процесса без изменения генерации','make check; make check-eval. Обучение и полная оценка не нужны.'],
        ['Изменение инструкции/декодирования','Синхронизировать контракт API и evaluation/params.json; make check; make release-reevaluate; создать новый workspace через src.release_bridge. До новой оценки старый допуск не действует.'],
        ['Новые данные обучения','uv run dvc repro tokenize; проверить схему/пересечения. При изменении обучающих примеров нужно новое обучение, затем полная оценка.'],
        ['Новый golden','Подготовить проверенный набор и полный индекс; uv run dvc repro eval_data; make release-reevaluate. Изменение только golden не требует обучения прежних весов.'],
        ['Новый адаптер','Закрепить его URI/version/run_id в metrics/candidate-model.json после проверки training run; make release-reevaluate; новый src.release_bridge workspace; lab.py run с manifest-candidate.json.']])
    report=f'''# ДЗ8+9+10 — процесс выпуска и настоящий API

## 1. Код и исходные условия

[Репозиторий](https://github.com/Rcinderblock/bmstu-mlops), исходный прототип `{ORIGINAL}`, исправленная реализация `{FIXED}`. [Полный diff](results/hw8910/fixes.patch), [семь причин](../defects.csv). Устройство: {hardware['cpu']}, {hardware['memory_bytes']/2**30:.0f} ГиБ, {hardware['device']}, {hardware['dtype']}; Python {hardware['python']}. Prefect 3.8.8, FastAPI 0.139.0, uvicorn 0.54.0. Torch/Transformers/PEFT закреплены в manifest и uv.lock. Совместимые версии проверены запуском; более новый FastAPI 0.161.1 оказался несовместим с Body(min_items) в этом Prefect, поэтому выбран и закреплён 0.139.0 без правки сторонних библиотек.

## 2. Семь причин и собственные проверки

{defect_table}

{test_table}

В собственном наборе три настоящих HTTP-опыта, три опыта с отдельными процессами и ещё один с реальным файлом и mock ошибки os.replace. Всего 7, из них 6 интеграционных, сверх 13 приложенных. На прототипе все семь дают FAIL из-за наблюдаемого неверного поведения; ошибки импорта/неизвестные команды не засчитываются. В тесте дубликатов исходник падает на двух событиях до дополнительной точки сбоя; в тесте версий — на ложном ID до дополнительной команды отката. [До](results/hw8910/tests-integration-before.txt), [после](results/hw8910/tests-fast-after.txt), подробные пары журналов [before](results/hw8910/before) / [after](results/hw8910/after).

Приложенный test_release.py побайтно совпадает с оригиналом. Отпечаток по имени и содержимому **09bd65e6b886**, SHA-256 файла 03632e58fa10f8742a205ae04ecba956f7a7cd0a0cfb90fc7fa9d32d6f2c1d7e. Прежние проверки сохранены: 32 проверки оценки и 40 модульных прошли отдельно. Историческая техническая приёмка обучения относится к своему старому коммиту; она не объявляется заново пройденной на расширенном src/.

## 3. Что переиспользовано

Читается готовый run `e4f47194810e4c4daa7826b58f8fd88d`, версия `models:/planner-qwen-adapter/1`; повторного sweep/обучения нет. Полная оценка — `14364972e4be484e988fd9691fdd036f`, 54 golden + 5 behavioral. [Origin](results/hw8910/origin.json) связывает артефакт, зафиксированные версии библиотек и исходный Git-коммит. Хеш обучавшего кода проверен по историческому коммиту, а не по новым модулям API.

Принятая версия — исходная `Qwen/Qwen3-0.6B`, revision c1899de289a04d12100db370d81485cdf75e47ca, model_id `{budget['model_id']}`. Она не обучалась нами: её веса получены из закреплённого источника Hugging Face. Поле source_training_run_id связывает общее описание базы с обучающим опытом, а не приписывает baseline локальное обучение. Кандидат остаётся отклонённым: 21/54 против 27/54 у базы. Registry alias champion не менялся; accepted.json — отдельный локальный указатель.

## 4. Граф, входы и восстановление

```mermaid
flowchart LR
    I[Вход и параметры] --> P[Проверка схемы / постоянный кеш]
    P --> A[Чтение готового артефакта / хеши файлов]
    A --> G[Полные отчёты / прежний quality_gate]
    G -->|отказ| R[Отчёт / прежний указатель]
    G -->|PASS| S[SQLite intent / accepted.json / событие]
    S --> API[Загрузка и прогрев API]
```

Четыре отдельные задачи в `src/release_flow.py`; между ними передаются описания URI/run ID/хешей, не веса или тензоры. Задача подготовки действительно читает и проверяет 54 JSONL-вопроса. Ключ включает SHA-256 байтов, tokenizer_revision, template_sha256, max_length, code_revision и фактический SHA кода обработки. Порядок dict и путь не заменяют содержимое. Prefect добавляет отдельное пространство хранения workspace, чтобы не ссылаться на уже удалённый каталог чужого опыта. Результаты — JSON в cache/ и .prefect-results/. [Настройки задач и постоянных результатов](https://reference.prefect.io/prefect/tasks/) соответствуют использованному интерфейсу.

{flow_table}

[Команды, exit codes, run IDs и SHA до/после](results/hw8910/flow-experiments/results.json), [все состояния задач](results/hw8910/attempts.jsonl). После управляемого сбоя следующий процесс получает Cached. Изменённые байты по тому же пути вызывают новую подготовку; изменённый смысл golden дополнительно отклоняется как несовместимый со старой оценкой. Временная ConnectionError/TimeoutError при чтении артефакта допускает максимум два повтора; неверная схема, веса и отказ gate не повторяются.

## 5. Публикация и согласованность

accepted.json пишется временным файлом в той же папке, flush/fsync выполняются до os.replace, при ошибке temp удаляется. Отказ не вызывает замену указателя. Во всех реальных отрицательных опытах SHA до и после одинаков: `fe713a8644049fc354fc944fb86654d0b39d307d8d0fc8c1e345af30cfa8a619`.

SQLite — первичный журнал уникальных выпусков. release_id учитывает model_id, model_sha256, runtime_sha256 и SHA связанного решения. До замены указателя сохраняется pending intent; после — committed. При сбое между ними повтор сверяет указатель с intent и завершает событие. publications.jsonl — восстанавливаемая проекция SQLite. attempts.jsonl хранит все попытки и состояния задач отдельно. [Реальный журнал](results/hw8910/publications.jsonl) содержит одно событие, хотя успешный выпуск повторялся в другом процессе. [Опыт сбоя/восстановления STUB](results/hw8910/after/test_identical_publication_and_repair_after_pointer_crash.jsonl) подтверждает отсутствие двойного события. Это не распределённая транзакция и не защита от нескольких писателей или всех вариантов потери питания.

## 6. Полный gate и привязка runtime

[Manifest базы](results/hw8910/manifest-baseline.json), [кандидата](results/hw8910/manifest-candidate.json), [решение базы](results/hw8910/release-report-baseline.json), [отказ кандидата](results/hw8910/release-report-candidate.json). Обвязка сверяет SHA полных исходных отчётов и сырых ответов, заново пересчитывает per-item/срезы/behavioral, вызывает неизменённое правило качества. PASS не записывается вручную и не заменяется одним val_loss. Для сохранения baseline правило сравнивает её полную оценку с ней же; это сохранение принятой исходной версии, не заявление об улучшении.

Runtime фиксирует точную базу, токенизатор, шаблон, instruction, seed, do_sample=False, thinking=False, лимит 32, bfloat16, MPS и версии библиотек генератора. Экспорт шаблона из tokenizer_config в отдельный файл не изменил ни одного из 59 текстов чата или наборов token IDs — [проверка](results/hw8910/tokenizer-equivalence.json). Условия генерации остались прежними, поэтому новая полная генерация не требовалась. Хеши фактических файлов пересчитываются перед публикацией и перед загрузкой; runtime_sha256 пересчитывается из всего словаря. Неполный/чужой отчёт, строковый PASS, NaN и ошибка замены имеют отдельные доказательства. STUB ONLY явно синтетичен и не считается настоящей оценкой.

## 7. Быстрый запуск и намеренная поломка

`make check` запускает 20 проверок. Изолированный fast-env создан из requirements-fast.txt: в нём отсутствуют torch, transformers, mlflow и веса; при этом HTTP, файловая система, Prefect и subprocess настоящие. `make check-negative` запускает те же контрактные проверки на временной копии с намеренно испорченным cache_key и возвращает ненулевой код; рабочий файл не изменён. [Журнал поломки](results/hw8910/tests-intentional-failure.txt). Исходники восстановлены, зелёный набор прошёл. Проверки выполнены локально; удалённый Actions-run и защита merge не заявляются.

{methods}

При смене модели/данных/условий нужны новые отчёты и новый workspace. Например: `make release-reevaluate`, затем `uv run python -m src.release_bridge --workspace deployments/new --reports-dir metrics` и `uv run python lab.py --workspace deployments/new run --manifest manifest-candidate.json --report release-report-candidate.json`. Изменение поддерживаемого лимита требует согласованно изменить контракт API и тесты; произвольный параметр запроса не наследует старый допуск. Полная оценка — генерация модели, а не быстрый CI.

## 8. Настоящий API

Ресурсы загружаются один раз в управляемом [lifespan FastAPI](https://fastapi.tiangolo.com/advanced/events/). Фоновая загрузка проверяет связанный отчёт и файлы, затем загружает Transformers/PEFT и делает настоящее вычисление для прогрева. До этого ready и generate возвращают 503; health подтверждает только жизнь процесса. Валидация ограничивает длину и строгие типы, допускается только оценённый лимит 32. Одновременных вычислений максимум одно. /debug включён только для STUB или явного локального --debug.

{readiness}

Настоящий ответ на «поставь будильник на семь утра»:

```json
{json.dumps(smoke,ensure_ascii=False,indent=2)}
```

[Полная последовательность жизни, неверных запросов и повторной загрузки](results/hw8910/api/lifecycle.json). Неверные запросы получили 422, вызовов генератора до/после 0/0. Это ответ Transformers, не STUB. После перезапуска загружена та же принятая версия с тем же runtime_sha256.

## 9. Загрузка версии и откат

[Отдельный HTTP-опыт A → B → A](results/hw8910/after/test_http_identity_restart_and_rollback.jsonl) использует две лёгкие STUB-версии, настоящие процессы и реальные файлы. Публикация B не меняет загруженную A; перезапуск загружает B. Откат разрешён только к ранее принятой A с проверенным отчётом; до перезапуска память остаётся B, затем ответ снова A. Это инженерный опыт, не новая настоящая модель. Реальная загрузка baseline доказана отдельно выше.

## 10. Замер 8+8

{batch_table}

[Все 16 попыток CSV](load-results.csv), [неизменённые ответы JSONL](results/hw8910/api/attempts.jsonl), [сырые итоги](results/hw8910/api/summary.json). Один и тот же набор, лимит 32, настоящий прогрев, одинаковые model_id/runtime и даже тексты/число токенов в двух режимах. Ошибки/тайм-ауты сохраняются; в этом запуске их не было, поэтому задержки ошибок обозначены null, а не выдуманным нулём. Успех в таблице означает **HTTP 200**, не правильную метку. Например, alarm-query получил обрезанный ответ в 32 токена, calendar-query — ошибочную alarm_query. Качество не маскируется транспортной успешностью.

![Замер](load-results.png)

Последовательный вычислитель объясняет почти неизменные 2.29 запроса/с и рост клиентской задержки: примерно 0.395 с среднего ожидания добавляются к 0.434 с вычисления. Клиентская задержка дополнительно включает HTTP, сериализацию и планирование потоков. Приведены средняя и максимум, не устойчивый p95. Один пакет и один прогрев не изолируют шум ОС, кэш и форму запросов; статистическая значимость не заявляется.

Prefill обрабатывает входную последовательность, decode добавляет токены последовательно. KV-cache хранит ключи/значения внимания уже обработанных токенов; размер растёт с контекстом и числом одновременно вычисляемых последовательностей. Для конфигурации этой модели теоретически {kv_bytes:,} байт на токен/последовательность в bfloat16, без служебных расходов; это расчёт по config.json, не измеренный максимум. Очередь HTTP не равна батчу вычислений. Ответ отдаётся одним JSON, поэтому TTFT и ITL не измерены. Этот Transformers-сервис с semaphore=1 не проверяет преимущества планировщиков, совместных батчей и управления памятью SGLang/vLLM.

## 11. Доступ и видео

На GitHub — код, настройки, тесты, отчёты и маленькие журналы. Веса/окружения/кеши/тома/лекции/голос исключены. [Текстовые входы и доказательства](https://github.com/Rcinderblock/bmstu-mlops/releases/tag/release-inputs-v1) публикуются отдельно с закреплённым SHA-256. `make release-restore` восстанавливает их и скачивает принятую базу из Hugging Face по точной revision, проверяя фактические хеши. Доступ к исходным полным отчётам также есть в docs/results/hw67/. Веса отклонённого адаптера в приложение не включаются и для обслуживания baseline не нужны; локальные опыты кандидата используют сохранённый MLflow-артефакт.

Одно видео 5–8 минут готовится локально и передаётся отдельно от Git. Памятка и команда отправки будут сохранены в notes/submission/hw08-09-10/README.md после проверки итогового MP4.
'''
    (ROOT/'docs/conclusion.md').write_text(report)
    (ROOT/'docs/defects.md').write_text('# Семь независимых причин\n\n'+defect_table+'\n\n[Полная таблица](../defects.csv), [исправления](results/hw8910/fixes.patch), [красный прогон](results/hw8910/tests-integration-before.txt), [зелёный прогон](results/hw8910/tests-fast-after.txt). Новые тесты не подменяют модель; настоящий API проверен отдельно.\n')
    overview=f'''# ДЗ8+9+10 — обзор результатов

## Задание

Семь причин → собственные проверки до/после → Prefect → полный gate → настоящий FastAPI → 8+8 запросов. Исходная модель принята; ухудшившийся кандидат отклонён.

## Дефекты

{defect_table}

## Проверки

{test_table}

20 = 13 исходных + 7 своих. Из своих 6 интеграционных: 3 HTTP + 3 subprocess. Отдельное окружение без torch/transformers/mlflow/весов. Намеренная поломка копии: ненулевой exit; рабочая версия не изменена. Отпечаток 09bd65e6b886.

## Процесс

Подготовка → готовый артефакт → связанный gate → публикация. Веса остаются файлами, задачи передают описания. Нового обучения нет.

{flow_table}

## Кеш

Байты входа + tokenizer_revision + template_sha256 + max_length + code_revision + фактический SHA обработки. Результаты на диске. Новый процесс: Cached. Изменение вопроса по прежнему пути: PREPARED и отказ из-за несовместимой оценки.

## Публикация

accepted.json: temp → flush/fsync → os.replace. attempts.jsonl: все попытки. SQLite: уникальные выпуски, pending/committed. publications.jsonl: восстанавливаемая проекция. Один писатель. Один успешный выпуск baseline, повтор не добавил событие. Сбой после указателя проверен на STUB и восстановлен.

## Допуск

| Модель | Верно / 54 | Формат / 54 | Поведение / 5 | Решение |
|---|---|---|---|---|
| Исходная | 27 | 51 | 5 | Сохранить исходную |
| Кандидат | 21 | 54 | 4 | Отказать |

Полные исходные отчёты и raw проверяются по SHA и пересчитываются. PASS вручную не задаётся. Принятая база: `{budget['model_id']}`. Runtime SHA: `{budget['runtime_sha256']}`. Повторная генерация полной оценки не требовалась: условия прежние; 59/59 текстов чата и token IDs совпали.

## Готовность

{readiness}

Жизнь процесса ≠ загруженная и прогретая модель. Генерация до ready: 503. Веса выбираются по manifest, не params.yaml.

## API

```json
{json.dumps(smoke,ensure_ascii=False,indent=2)}
```

Реальный Transformers, bfloat16, MPS. Лимит 32. Неправильные запросы: 422, calls 0 → 0. Один вычислитель.

## Версии

STUB ONLY: загрузить A → опубликовать B → ответ всё ещё A → перезапустить → B → откатить к принятой A → до перезапуска B → после перезапуска A. Настоящая база отдельно загружена повторно с тем же ID/runtime.

## Нагрузка

{batch_table}

HTTP 200 — транспортный успех, не правильная метка. Все 16 строк и ответы сохранены; выводы о качестве не подменяются скоростью.

![Нагрузка](load-results.png)

## Репозиторий

Код: release.py, lab.py, service.py, src/release_flow.py, src/api_backend.py. Отчёты: defects.csv, docs/conclusion.md, docs/load-results.csv, docs/results/hw8910/. Только локальный CI. Восстановление принятой базы: make release-restore. Веса и лекции в Git не входят.

Не измерялись устойчивый p95, TTFT, ITL и SGLang/vLLM. Prefill обрабатывает вход, decode добавляет токены. KV-cache растёт с контекстом и вычисляемыми последовательностями. Два HTTP-запроса здесь не превращаются в вычислительный батч.

## Итог

20 быстрых проверок; семь причин воспроизведены и исправлены. Сбой/кеш/повтор/откат проверены. Настоящий API обслуживает проверенную исходную версию. Кандидат отклонён, пороги не ослаблялись. Все 16 попыток сохранены. Спасибо за внимание.
'''
    (ROOT/'docs/overview.md').write_text(overview)
    readme=f'''# Личный планировщик — выпуск модели и HTTP API

ДЗ8+9+10: исправлены семь причин сбоев, выполнен Prefect-flow, подключён настоящий FastAPI с Qwen3-0.6B. **Кандидат отклонён по качеству; обслуживается исходная модель.** Пороги не ослаблялись.

- [Отчёт и требования](docs/conclusion.md), [семь причин](defects.csv), [исправления](docs/defects.md).
- [Красные проверки](docs/results/hw8910/tests-integration-before.txt) → [20 зелёных](docs/results/hw8910/tests-fast-after.txt), [намеренная поломка](docs/results/hw8910/tests-intentional-failure.txt).
- [Настоящий ответ и перезапуск](docs/results/hw8910/api/lifecycle.json), [принятое описание](docs/results/hw8910/accepted.json), [связанный допуск](docs/results/hw8910/release-report-baseline.json).
- [Все 16 попыток](docs/load-results.csv), [сравнение нагрузки](docs/load-results.png): 8/8 HTTP 200 в каждом режиме; около 2.29 запроса/с; средняя задержка 0.436 → 0.834 с. Это успех HTTP, не утверждение о качестве меток.
- Исходный коммит `{ORIGINAL}`; исправленный `{FIXED}`. Тесты преподавателя сохранены, отпечаток **09bd65e6b886**.

## Запуск на новом ноутбуке

Нужны Python 3.12 и uv. Для модели — память под Qwen3-0.6B и устройство MPS с bfloat16, как в сохранённой оценке. На другом устройстве/типе чисел нужна новая полная оценка; автоматическая подмена runtime не допускается.

```sh
make install
make check                 # 20 проверок без загрузки весов
make check-negative        # намеренная поломка копии; ожидается ненулевой код
make release-restore       # новый deployments/planner; тексты + база HF по revision
make release               # полный связанный gate → accepted.json
make api                   # localhost:8091; отдельный терминал для запросов
```

```sh
curl -s http://127.0.0.1:8091/health
curl -s http://127.0.0.1:8091/ready
curl -s http://127.0.0.1:8091/generate -H 'Content-Type: application/json' -d '{{"prompt":"поставь будильник на семь утра","max_new_tokens":32}}'
```

До загрузки и прогрева ready/generate возвращают 503. Лимит зафиксирован оценкой: 32; длина и типы проверяются до вычислений. В ответе — фактически загруженная версия и runtime SHA. Изменение accepted.json не меняет модель в памяти; нужен управляемый перезапуск.

Для быстрых проверок без библиотек модели:

```sh
uv venv --python 3.12 .venv-fast
uv pip install --python .venv-fast/bin/python -r requirements-fast.txt
.venv-fast/bin/python scripts/check_release.py --assert-lightweight
```

Prefect запускается локально автоматически на время команды, если сервер ещё не работает. `make prefect` позволяет оставить его открытым на localhost:4208. Результаты и metadata хранятся в локальных каталогах; облачный аккаунт не нужен. Удалённый CI и защита merge не заявляются.

## Где что хранится

`release.py` — ключ/контракт/атомарный JSON. `src/deployment.py` — проверка файлов и согласованный журнал. `src/release_flow.py` — четыре задачи. `service.py` и `src/api_backend.py` — готовность, валидация и реальный генератор. `deployments/planner` содержит файлы модели, неизменяемые доказательства, кеш, accepted.json и журнал; в Git он исключён.

[Приложение](https://github.com/Rcinderblock/bmstu-mlops/releases/tag/release-inputs-v1) содержит только тексты, полные отчёты, manifest и контрольные суммы. Принятые веса восстанавливаются отдельно из Hugging Face по revision. Веса отклонённого адаптера не публикуются: полный отказ и исходные ответы доступны в отчётах. Для локального стенда с сохранённым MLflow-артефактом `make release-init` читает существующий run без обучения; на новом ноутбуке используйте `make release-restore`.

`make api-bench` запускает самостоятельный опыт и останавливает свои серверы; порт 8091 перед этим должен быть свободен. Все попытки остаются в новом каталоге notes/work. `make check-eval` и `make test-legacy` сохраняют прежние модульные проверки. Исторические команды обучения — в conf/history/, предыдущие результаты — в docs/hw67-* и истории коммитов. Старую техническую приёмку запускают на её историческом коммите, поскольку общий src развивается.

Обучающий источник Amazon MASSIVE 1.0, [CC BY 4.0](docs/MASSIVE-LICENSE.txt). Проверочные команды разобраны с помощью агента; человеческая проверка не заявлена. Лекции, окружение, кеши, веса, голос и рабочие записи исключены из Git. Согласование темы преподавателем не зафиксировано.
'''
    (ROOT/'README.md').write_text(readme)
    print('Exported evidence and report from actual runs:',OUT)


if __name__=='__main__':main()
