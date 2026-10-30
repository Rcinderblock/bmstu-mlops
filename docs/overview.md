# ДЗ8+9+10 — обзор результатов

## Задание

Семь причин → собственные проверки до/после → Prefect → полный gate → настоящий FastAPI → 8+8 запросов. Исходная модель принята; ухудшившийся кандидат отклонён.

## Дефекты

| № | Причина | Исправление |
|---|---|---|
| 1 | Ключ хешировал только путь | SHA-256 входа и всех параметров плюс версия обработки; постоянный результат |
| 2 | Проверка истинности вместо полного контракта и связанного допуска | Строгий контракт и повторный вызов прежнего quality_gate по проверенным полным отчётам |
| 3 | Прямая перезапись accepted.json | Временный файл рядом, flush/fsync, os.replace, уборка при ошибке |
| 4 | Безусловный append без устойчивого release_id | SQLite: уникальный release_id и pending/committed; восстановление JSONL-проекции |
| 5 | ready=True до загрузки/прогрева | Фоновая загрузка и настоящий прогрев; независимые health/ready |
| 6 | Чтение указателя при каждом ответе | ID/runtime из загруженного backend; управляемый перезапуск и проверенный откат |
| 7 | Нет строгой валидации до вычисления | Строгие типы, длина, фиксированный оценённый лимит до вызова backend |

## Проверки

| Проверка | До | После |
|---|---|---|
| test_cache_recovery_and_changed_bytes_in_new_process | FAIL | ok |
| test_incomplete_and_foreign_reports_preserve_pointer | FAIL | ok |
| test_atomic_failure_preserves_real_file_bytes | FAIL | ok |
| test_identical_publication_and_repair_after_pointer_crash | FAIL | ok |
| test_http_readiness_waits_for_warmup | FAIL | ok |
| test_http_identity_restart_and_rollback | FAIL | ok |
| test_http_validation_does_not_call_backend | FAIL | ok |

20 = 13 исходных + 7 своих. Из своих 6 интеграционных: 3 HTTP + 3 subprocess. Отдельное окружение без torch/transformers/mlflow/весов. Намеренная поломка копии: ненулевой exit; рабочая версия не изменена. Отпечаток 09bd65e6b886.

## Процесс

Подготовка → готовый артефакт → связанный gate → публикация. Веса остаются файлами, задачи передают описания. Нового обучения нет.

| Опыт | Exit | Подготовка | Прежний указатель при отказе |
|---|---|---|---|
| 01-fail-after-prepare | 1 | PREPARED | да |
| 02-resume-new-process | 0 | CACHE_HIT | — |
| 03-idempotent-repeat | 0 | CACHE_HIT | — |
| 04-change-input-same-path | 1 | PREPARED | да |
| 05-candidate-quality-rejection | 1 | CACHE_HIT | да |
| 06-incomplete | 1 | CACHE_HIT | да |
| 07-truthy-string | 1 | CACHE_HIT | да |
| 08-foreign | 1 | CACHE_HIT | да |
| 09-nan | 1 | CACHE_HIT | да |

## Кеш

Байты входа + tokenizer_revision + template_sha256 + max_length + code_revision + фактический SHA обработки. Результаты на диске. Новый процесс: Cached. Изменение вопроса по прежнему пути: PREPARED и отказ из-за несовместимой оценки.

Выдержки из сохранённых запусков, полные журналы — `docs/results/hw8910/flow-experiments/`:

```text
01-fail-after-prepare
PREPARED eedfd8a3ad55062d023e00d8d2c48db58447054f6bcae99cec96d318c9b8e446
PREPARE_TASK_STATE Completed

02-resume-new-process
PREPARE_TASK_STATE Cached
CACHE_HIT eedfd8a3ad55062d023e00d8d2c48db58447054f6bcae99cec96d318c9b8e446

04-change-input-same-path
PREPARED c017feea0df89ac417f9a46317fec9089f89d38f480270260674b4674e436954
PREPARE_TASK_STATE Completed
```

## Публикация

accepted.json: temp → flush/fsync → os.replace. attempts.jsonl: все попытки. SQLite: уникальные выпуски, pending/committed. publications.jsonl: восстанавливаемая проекция. Один писатель. Один успешный выпуск baseline, повтор не добавил событие. Сбой после указателя проверен на STUB и восстановлен.

```text
STUB ONLY — сохранённый опыт в отдельном процессе
FLOW_RUN_ID fb8f1934-318c-43d6-9a93-0e12b618c288
RECOVERED_PUBLICATION 55c008643a124c52044ab2a2ca815a85982390552aa2106d7675104331776afb
PREPARE_TASK_STATE Cached
CACHE_HIT 88bbd559a04a7a432693a1c611ef2704bb711ac17dd5e22203ee9e5020f59fd4
PUBLISHED fixture-B 55c008643a124c52044ab2a2ca815a85982390552aa2106d7675104331776afb

число восстановленных событий: 1
model_id: fixture-B
```

## Допуск

| Модель | Верно / 54 | Формат / 54 | Поведение / 5 | Решение |
|---|---|---|---|---|
| Исходная | 27 | 51 | 5 | Сохранить исходную |
| Кандидат | 21 | 54 | 4 | Отказать |

Полные исходные отчёты и raw проверяются по SHA и пересчитываются. PASS вручную не задаётся. Принятая база: `planner-baseline-c1899de289a0`. Runtime SHA: `1c6c5cd2f5c9a3688e81cc8a55b4e251975b93fcc1c1b0649c6d6ed38e4a49c4`. Повторная генерация полной оценки не требовалась: условия прежние; 59/59 текстов чата и token IDs совпали.

## Готовность

| Событие | HTTP | Фактическая версия |
|---|---|---|
| initial-health | 200 | — |
| initial-not-ready | 503 | — |
| initial-ready | 200 | planner-baseline-c1899de289a0 |
| restart-ready | 200 | planner-baseline-c1899de289a0 |
| restart-real-smoke | 200 | planner-baseline-c1899de289a0 |

Жизнь процесса ≠ загруженная и прогретая модель. Генерация до ready: 503. Веса выбираются по manifest, не params.yaml.

## API

```json
{
  "answer": "{\"answer\": \"alarm_set\"}",
  "output_tokens": 8,
  "input_tokens": 261,
  "generation_seconds": 0.40936333301942796,
  "model_id": "planner-baseline-c1899de289a0",
  "runtime_sha256": "1c6c5cd2f5c9a3688e81cc8a55b4e251975b93fcc1c1b0649c6d6ed38e4a49c4",
  "request_id": "7bfdf05b-58a8-4cdc-b6fd-1d6318f63b68",
  "queue_seconds": 1.2208009138703346e-05,
  "service_seconds": 0.4126095420215279,
  "backend": "transformers"
}
```

Реальный Transformers, bfloat16, MPS. Лимит 32. Неправильные запросы: 422, calls 0 → 0. Один вычислитель.

## Версии

STUB ONLY: загрузить A → опубликовать B → ответ всё ещё A → перезапустить → B → откатить к принятой A → до перезапуска B → после перезапуска A. Настоящая база отдельно загружена повторно с тем же ID/runtime.

Сохранённые реальные HTTP-ответы стенда `STUB ONLY`:

| Событие | HTTP | ID из памяти | Номер запроса |
|---|---|---|---|
| После загрузки A | 200 | fixture-A | c168925e-d427-44b6-b9ef-c4c3679eeed2 |
| После публикации B, до перезапуска | 200 | fixture-A | aeeaca6e-570a-4387-9961-5d58fc0f65e1 |
| После перезапуска | 200 | fixture-B | 4d7c298f-cf03-4fcb-b76f-16c662324d16 |
| После отката к A, до перезапуска | 200 | fixture-B | bca1caf3-5c7b-4475-ab4b-d0f88e9a5869 |
| После следующего перезапуска | 200 | fixture-A | a4f100be-2c7e-4789-b2fd-6a224e041ba0 |

## Нагрузка

| Параллельность | n | HTTP 200 | Ошибок | Весь пакет, с | HTTP 200/с | Средняя, с | Максимальная, с | Очередь, с | Вычисление, с | Выходных токенов |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 8 | 0 | 3.490 | 2.292 | 0.436 | 0.977 | 0.000 | 0.434 | 94 |
| 2 | 8 | 8 | 0 | 3.491 | 2.291 | 0.834 | 1.293 | 0.395 | 0.434 | 94 |

HTTP 200 — транспортный успех, не правильная метка. Все 16 строк и ответы сохранены; выводы о качестве не подменяются скоростью.

![Нагрузка](load-results.png)

## Репозиторий

Код: release.py, lab.py, service.py, src/release_flow.py, src/api_backend.py. Отчёты: defects.csv, docs/conclusion.md, docs/load-results.csv, docs/results/hw8910/. Только локальный CI. Восстановление принятой базы: make release-restore. Веса и лекции в Git не входят.

Не измерялись устойчивый p95, TTFT, ITL и SGLang/vLLM. Prefill обрабатывает вход, decode добавляет токены. KV-cache растёт с контекстом и вычисляемыми последовательностями. Два HTTP-запроса здесь не превращаются в вычислительный батч.

## Итог

20 быстрых проверок; семь причин воспроизведены и исправлены. Сбой/кеш/повтор/откат проверены. Настоящий API обслуживает проверенную исходную версию. Кандидат отклонён, пороги не ослаблялись. Все 16 попыток сохранены. Спасибо за внимание.
