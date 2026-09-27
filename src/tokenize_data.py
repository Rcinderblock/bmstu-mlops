"""JSONL -> числа для обучения, измерения и отчёт. Загружается только токенизатор."""
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.collate import DynamicPaddingCollator, LABEL_PAD_ID
from src.config import load_params
from src.prompt import build_chat_text, prompt_token_len

METRICS_PATH = Path('metrics/tokenize.json')
REPORT_PATH = Path('docs/tokenize_report.md')


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'Пустая выборка: {path}')
    return rows


def mask_prompt(input_ids: list[int], n_prompt: int) -> list[int]:
    """Не меняем вход модели; исключаем промпт только из целевых меток."""
    n = min(n_prompt, len(input_ids))
    return [LABEL_PAD_ID] * n + list(input_ids[n:])


def encode_example(tokenizer, record: dict, params: dict, max_seq_len: int) -> dict:
    if max_seq_len <= 0:
        raise ValueError('max_seq_len должен быть положительным')
    messages = record['messages']
    full_text = build_chat_text(tokenizer, messages, params, add_generation_prompt=False)
    prompt = build_chat_text(tokenizer, messages, params, add_generation_prompt=True)
    if not full_text.encode('utf-8').startswith(prompt.encode('utf-8')):
        raise ValueError(f"{record.get('id')}: шаблоны обучения и генерации разошлись")
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    n_prompt, fallback = prompt_token_len(tokenizer, prompt, ids, encoded['offset_mapping'])
    # Ответ заканчивается первым EOS после промпта. Переводы строк после него
    # остаются во входе, но не становятся отдельной целью обучения.
    end = ids.index(tokenizer.eos_token_id, n_prompt) + 1 if n_prompt < len(ids) else len(ids)
    input_ids = ids[:max_seq_len]
    labels = mask_prompt(input_ids, n_prompt)
    labels[end:] = [LABEL_PAD_ID] * max(0, len(labels) - end)
    supervised = sum(x != LABEL_PAD_ID for x in labels)
    return {
        'id': record.get('id'), 'input_ids': input_ids,
        'attention_mask': [1] * len(input_ids), 'labels': labels,
        '_meta': {
            'id': record.get('id'), 'full_len': len(ids), 'prompt_len': n_prompt,
            'answer_len': end - n_prompt, 'truncated': len(ids) > max_seq_len,
            'bpe_fallback': fallback, 'supervised': supervised,
        },
    }


def describe(values: list[int]) -> dict:
    a = np.asarray(values)
    return {'count': int(a.size), 'mean': round(float(a.mean()), 2),
            **{f'p{q}': int(np.percentile(a, q)) for q in (50, 90, 99)}, 'max': int(a.max())}


def truncation_stats(metas: list[dict], name: str, params: dict) -> dict:
    count = sum(m['truncated'] for m in metas)
    ratio = count / len(metas) if metas else 0.0
    limit = params['tokenize']['truncated_warn_ratio']
    if ratio > limit:
        warnings.warn(f'{name}: обрезано {count}/{len(metas)} ({ratio:.1%}), порог {limit:.1%}', stacklevel=2)
    return {'truncated': count, 'truncated_ratio': ratio, 'truncation_pass': ratio <= limit}


def padding_stats(examples: list[dict], tokenizer, params: dict) -> dict:
    cfg = params['tokenize']
    collate = DynamicPaddingCollator(tokenizer.pad_token_id, tokenizer.padding_side)
    batches = [collate(examples[i:i + cfg['batch_size']]) for i in range(0, len(examples), cfg['batch_size'])]
    slots = sum(b['input_ids'].numel() for b in batches)
    tokens = sum(len(e['input_ids']) for e in examples)
    static = len(examples) * cfg['max_seq_len']
    first = batches[0]
    return {'batch_size': cfg['batch_size'], 'batches': len(batches),
            'dynamic_slots': slots, 'dynamic_pad_tokens': slots - tokens,
            'dynamic_pad_ratio': (slots - tokens) / slots,
            'static_slots': static, 'static_pad_ratio': (static - tokens) / static,
            'first_batch_shape': list(first['input_ids'].shape),
            'first_batch_lengths': [len(e['input_ids']) for e in examples[:cfg['batch_size']]],
            'padding_label_errors': sum(int((b['labels'][b['attention_mask'] == 0] != LABEL_PAD_ID).sum()) for b in batches)}


def process_split(tokenizer, name: str, path: Path, params: dict) -> tuple[list[dict], dict]:
    records = read_jsonl(path)
    examples, metas = [], []
    for row in records:
        encoded = encode_example(tokenizer, row, params, params['tokenize']['max_seq_len'])
        meta = encoded.pop('_meta')
        metas.append(meta)  # До отбрасывания: знаменатель включает все входные записи.
        if meta['supervised']:
            examples.append(encoded)
    stats = {
        'examples_in': len(records), 'examples_kept': len(examples),
        'dropped_no_supervision': len(records) - len(examples),
        'length_tokens': describe([m['full_len'] for m in metas]),
        'prompt_tokens': describe([m['prompt_len'] for m in metas]),
        'answer_tokens': describe([m['answer_len'] for m in metas]),
        'bpe_boundary_fallback': sum(m['bpe_fallback'] for m in metas),
        'template_prefix_matches': len(metas),
        'total_tokens': sum(len(e['input_ids']) for e in examples),
        'supervised_tokens': sum(m['supervised'] for m in metas),
        **truncation_stats(metas, name, params),
    }
    stats['padding'] = padding_stats(examples, tokenizer, params) if examples else None
    stats['supervised_ratio'] = stats['supervised_tokens'] / stats['total_tokens'] if examples else 0
    print(f"{name}: {len(examples)}/{len(records)} примеров; {stats['total_tokens']} токенов; "
          f"в лосс {stats['supervised_tokens']}; обрезано {stats['truncated']} ({stats['truncated_ratio']:.1%})")
    return examples, stats


def estimate_train_time(total_tokens: int, params: dict) -> dict:
    cfg = params['train_estimate']
    benchmark = json.loads(Path(cfg['generation_benchmark']).read_text())
    tps = benchmark['tokens_per_sec']
    if benchmark['model'] != params['model']['name'] or tps <= 0:
        raise ValueError('Замер скорости относится к другой модели или некорректен')
    low, high = cfg['slowdown_min'], cfg['slowdown_max']
    if not 0 < low <= high or cfg['epochs'] <= 0:
        raise ValueError('Некорректные предположения о времени обучения')
    seconds = total_tokens * cfg['epochs'] / tps
    return {'epochs': cfg['epochs'], 'tokens_per_epoch': total_tokens,
            'generation_tokens_per_sec': tps, 'source': cfg['generation_benchmark'],
            'assumed_training_tokens_per_sec': [tps / high, tps / low],
            'seconds_range': [seconds * low, seconds * high],
            'hours_range': [seconds * low / 3600, seconds * high / 3600],
            'note': 'Оценка при замедлении в 2–3 раза относительно измеренной генерации. '
                    'Обучение не запускалось. Скорости генерации и обучения напрямую не сопоставимы; '
                    'это учебный сценарий, а не измерение или гарантированный диапазон. '
                    'Считаются все входные токены, включая замаскированный промпт, без паддинга.'}


def render_report(m: dict) -> str:
    train = m['splits']['train']; pad = train['padding']; est = m['train_time_estimate']
    lines = ['# ДЗ4 — подготовка последовательностей для обучения', '',
             'Сгенерировано стадией `tokenize`. Этот файл не правится вручную.', '',
             '## Токены и длины', '',
             f"Модель: `{m['model']}`. Предел: **{m['max_seq_len']} токенов**. Загружен только токенизатор.", '',
             '| выборка | примеров | p50 | p90 | p99 | максимум | обрезано | всего токенов | в лосс |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, s in m['splits'].items():
        q = s['length_tokens']
        lines.append(f"| {name} | {s['examples_kept']} | {q['p50']} | {q['p90']} | {q['p99']} | {q['max']} | "
                     f"{s['truncated']} ({s['truncated_ratio']:.1%}) | {s['total_tokens']} | {s['supervised_tokens']} |")
    lines += ['', 'Токен — элемент словаря токенизатора; он может соответствовать слову, части слова или служебной границе.', '',
              '## Шаблон чата', '', '`src/prompt.py` — общее место сборки диалога для подготовки данных и генерации.', '',
              f"`enable_thinking={str(m['enable_thinking']).lower()}`. Проверено префиксов train / val: "
              f"**{train['template_prefix_matches']} / {m['splits']['val']['template_prefix_matches']}**.", '',
              'Промпт генерации побайтово совпадает с началом полного диалога. Роли и служебные границы берутся из шаблона токенизатора.', '',
              '## Маска лосса', '',
              f"Обучающих позиций: **{train['supervised_tokens']} / {train['total_tokens']} ({train['supervised_ratio']:.2%})**.", '',
              '`input_ids` сохраняет весь контекст. В `labels` инструкция, вопрос и пустой блок рассуждений заменены на `-100`; ответ и EOS остаются целями. Текст после EOS тоже исключён из лосса.', '',
              'Число -100 исключает позицию из расчёта ошибки, но не скрывает контекст от механизма внимания. Метки заранее не сдвигаются: сдвиг делает функция потерь модели.', '',
              f"Запасной путь по границам символов: train **{train['bpe_boundary_fallback']}**, val **{m['splits']['val']['bpe_boundary_fallback']}**. Если токен пересекает границу промпта и ответа, он маскируется.", '',
              '## Форма батча', '']
    if pad:
        lines += [f"Первый батч: **{pad['first_batch_shape'][0]} × {pad['first_batch_shape'][1]}**. Длины строк: `{pad['first_batch_lengths']}`.", '',
                  '| массив | настоящие позиции | добавленные слева позиции |',
                  '|---|---|---|', '| input_ids | токены диалога | pad_token_id |',
                  '| attention_mask | 1 | 0 |', '| labels | -100 либо токен ответа | -100 |', '',
                  f"Динамическое заполнение: **{pad['dynamic_pad_ratio']:.2%}** позиций. При заполнении до {m['max_seq_len']} всегда: **{pad['static_pad_ratio']:.2%}**. Ошибок в метках на заполнении: **{pad['padding_label_errors']}**.", '',
                  'Длина батча равна его самой длинной строке. Примеры не склеиваются между собой; необязательная упаковка отключена.', '']
    lines += ['## Обрезка', '', f"Порог: **{m['truncated_warn_ratio']:.1%}**. Доля считается по всем входным записям до удаления примеров без ответа.", '']
    for name, s in m['splits'].items():
        lines.append(f"- {name}: обрезано {s['truncated']}/{s['examples_in']}; без обучающих позиций удалено {s['dropped_no_supervision']}; порог {'пройден' if s['truncation_pass'] else 'ПРЕВЫШЕН'}.")
    lines += ['', 'Стадия завершается с ошибкой при превышении порога или пустом результате, сохраняя диагностический отчёт.', '',
              '## Прогноз времени обучения', '',
              f"Входных токенов за эпоху: **{est['tokens_per_epoch']}**, эпох: **{est['epochs']}**. Измеренная скорость генерации: **{est['generation_tokens_per_sec']:.2f} токена/с**.", '',
              f"При условной скорости обучения **{est['assumed_training_tokens_per_sec'][0]:.2f}–{est['assumed_training_tokens_per_sec'][1]:.2f} токена/с** получается **{est['hours_range'][0]:.2f}–{est['hours_range'][1]:.2f} часа**.", '', est['note'], '']
    return '\n'.join(lines)


def main() -> None:
    params = load_params()
    if params['packing']['enabled']:
        raise ValueError('Упаковка отключена: без блочной маски соседние примеры смешаются')
    cfg = params['tokenize']
    if cfg['padding_side'] != 'left' or cfg['batch_size'] <= 0 or not 0 <= cfg['truncated_warn_ratio'] <= 1:
        raise ValueError('Нужны левое заполнение, положительный размер батча и порог от 0 до 1')
    tokenizer = AutoTokenizer.from_pretrained(params['model']['name'])
    tokenizer.padding_side = cfg['padding_side']
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    out_dir = Path(params['data']['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name, key in (('train', 'train_jsonl'), ('val', 'val_jsonl')):
        examples, stats = process_split(tokenizer, name, Path(params['data'][key]), params)
        torch.save({'examples': examples, 'model': params['model']['name'],
                    'max_seq_len': cfg['max_seq_len'], 'padding_side': tokenizer.padding_side,
                    'pad_token_id': tokenizer.pad_token_id}, out_dir / f'{name}.pt')
        splits[name] = stats
    metrics = {'model': params['model']['name'], 'enable_thinking': params['model'].get('enable_thinking'),
               'max_seq_len': cfg['max_seq_len'], 'padding_side': tokenizer.padding_side,
               'truncated_warn_ratio': cfg['truncated_warn_ratio'], 'splits': splits,
               'train_time_estimate': estimate_train_time(splits['train']['total_tokens'], params)}
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(metrics), encoding='utf-8')
    if any(not s['truncation_pass'] or not s['examples_kept'] for s in splits.values()):
        raise SystemExit('Проверка обрезки не пройдена: см. metrics/tokenize.json')
    print(f'Готово: {out_dir}, {METRICS_PATH}, {REPORT_PATH}')


if __name__ == '__main__':
    main()
