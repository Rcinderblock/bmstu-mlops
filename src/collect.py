"""Создать корпус команд планировщика из русского MASSIVE с проверкой разметки."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from src.config import load_params
from src.schema import Example, dump
from src.textnorm import normalize_text

SLOT = re.compile(r'\[([^:\[\]]+)\s*:\s*([^\[\]]+)\]')


def template(annotation):
    """Все значения слотов заменены типами; одинаковые каркасы — одна группа."""
    return normalize_text(SLOT.sub(lambda m: '{' + m[1].strip() + '}', annotation))


def main():
    p = load_params()
    cfg, paths = p['collect'], p['paths']
    meta = json.loads(Path(cfg['source_manifest']).read_text())
    source = Path(cfg['sources'][cfg['version']][0])
    if hashlib.sha256(source.read_bytes()).hexdigest() != meta['file_sha256']:
        raise ValueError('Источник изменился: SHA-256 не совпадает')
    rows = [json.loads(x) for x in source.read_text().splitlines()]
    order = {name: i for i, name in enumerate(cfg['partitions']['v2'])}
    rows.sort(key=lambda r: (order[r['partition']], int(r['id'])))
    counts = Counter(scanned=len(rows))
    out = []
    seen = set()
    labels = cfg['labels']
    # Одинаковые запросы с противоречащими метками не выбираем случайно.
    # Реестр строится по всему фиксированному источнику одинаково для v1 и v2.
    answers = defaultdict(set)
    for r in rows:
        if (r['locale'] == cfg['locale'] and r['scenario'] in cfg['scenarios']
            and r['intent'] in labels
            and sum(j['intent_score'] == 1 and j['language_identification'] == 'target' for j in r['judgments']) >= cfg['min_intent_votes']):
            answers[normalize_text(r['utt'])].add(r['intent'])
    conflicts = {text for text, values in answers.items() if len(values) > 1}
    descriptions = '; '.join(f'{key}: {value}' for key, value in labels.items())
    for r in rows:
        if r['locale'] != cfg['locale'] or r['scenario'] not in cfg['scenarios']:
            counts['outside_domain'] += 1
            continue
        counts['domain_rows'] += 1
        if r['partition'] not in cfg['partitions'][cfg['version']]:
            counts['outside_version'] += 1
            continue
        if re.search(cfg['exclude_user_pattern'], r['utt']):
            counts['outside_task'] += 1
            continue
        if r['intent'] not in labels or not r['intent'].startswith(r['scenario'] + '_'):
            counts['invalid_label'] += 1
            continue
        votes = sum(j['intent_score'] == 1 and j['language_identification'] == 'target' for j in r['judgments'])
        if votes < cfg['min_intent_votes']:
            counts['rejected_votes'] += 1
            continue
        if normalize_text(r['utt']) in conflicts:
            counts['conflicting_label_rows'] += 1
            continue
        restored = SLOT.sub(lambda m: m[2].strip(), r['annot_utt'])
        if normalize_text(restored) != normalize_text(r['utt']):
            counts['annotation_mismatch'] += 1
            continue
        if r['id'] in seen:
            raise ValueError('Повтор source id: ' + r['id'])
        seen.add(r['id'])
        idx = int(hashlib.sha256(r['id'].encode()).hexdigest(), 16) % len(cfg['system_prompts'])
        # Каркас вычисляется до разбиения, без метки ответа. Хеш только сокращает его.
        group = hashlib.sha256(template(r['annot_utt']).encode()).hexdigest()[:20]
        example = Example.model_validate({
            'id': 'massive-ru-' + r['id'], 'topic': 'planner-template-' + group,
            'messages': [
                {'role': 'system', 'content': cfg['system_prompts'][idx] + '\nМетки: ' + descriptions},
                {'role': 'user', 'content': r['utt']},
                {'role': 'assistant', 'content': r['intent']},
            ],
        })
        out.append(example)
    target = Path(paths['raw']); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(''.join(dump(x) + '\n' for x in out))
    metrics = {'version': cfg['version'], **{k: counts[k] for k in ['scanned','outside_domain','domain_rows','outside_version','outside_task','invalid_label','rejected_votes','conflicting_label_rows','annotation_mismatch']},
               'written': len(out), 'labels': dict(sorted(Counter(x.assistant for x in out).items())),
               'source_sha256': meta['file_sha256']}
    dest = Path(paths['metrics_collect']); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')
    print(f"collect {cfg['version']}: {len(rows)} → {len(out)} строк; проверены метки, голоса аннотаторов и слоты")


if __name__ == '__main__':
    main()
