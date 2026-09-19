"""Стабильное разделение целых групп: добавление данных не переносит старые группы."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from src.config import load_params
from src.schema import dump, iter_examples
from src.textnorm import normalize_group


def group_label(group, ratios, seed):
    if set(ratios) != {'train','val','test'} or any(v <= 0 for v in ratios.values()) or not math.isclose(sum(ratios.values()), 1):
        raise ValueError('Нужны положительные доли train/val/test с суммой 1')
    value = int(hashlib.sha256(f'{seed}:{normalize_group(group)}'.encode()).hexdigest(), 16) / 2**256
    total = 0
    for name, ratio in ratios.items():
        total += ratio
        if value < total:
            return name
    return name


def main():
    p = load_params(); cfg, paths = p['split'], p['paths']
    if cfg['group_key'] != 'topic':
        raise ValueError('Поддерживается group_key=topic')
    rows = list(iter_examples(paths['clean']))
    buckets = {name: [] for name in cfg['ratios']}
    for ex in rows:
        buckets[group_label(ex.topic, cfg['ratios'], cfg['seed'])].append(ex)
    required = set(p['collect']['labels'])
    for name, examples in buckets.items():
        if not examples or {ex.assistant for ex in examples} != required:
            raise ValueError(f'{name}: отсутствуют примеры одного или нескольких классов')
        target = Path(paths[name]); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(''.join(dump(ex) + '\n' for ex in examples))
    metrics = {'version': p['collect']['version'], 'seed': cfg['seed'], 'group_key': cfg['group_key'],
               'groups_total': len({normalize_group(x.topic) for x in rows}),
               'sizes': {k: len(v) for k,v in buckets.items()},
               'groups': {k: len({normalize_group(x.topic) for x in v}) for k,v in buckets.items()},
               'ratios_actual': {k: round(len(v)/len(rows),4) for k,v in buckets.items()},
               'labels': {k: dict(sorted(Counter(x.assistant for x in v).items())) for k,v in buckets.items()}}
    Path(paths['metrics_split']).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')
    print('split: ' + ', '.join(f'{k}={len(v)}' for k,v in buckets.items()))


if __name__ == '__main__':
    main()
