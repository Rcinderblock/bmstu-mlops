"""Независимый гейт: все три пары выборок и единый порог сходства."""
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_params
from src.contamination import report, is_clean
from src.schema import iter_examples


def main():
    p = load_params(); nd = p['clean']['near_dup']
    if p['contamination']['threshold'] != nd['threshold']:
        raise ValueError('Пороги очистки и проверки должны совпадать')
    splits = {name: list(iter_examples(p['paths'][name])) for name in ['train','val','test']}
    pairs = {}
    for a,b in itertools.combinations(splits, 2):
        rep = report(splits[a], splits[b], nd['shingle_words'], nd['num_perm'], nd['threshold'])
        pairs[a + '/' + b] = {k:v for k,v in rep.items() if k != 'examples'}
        print(f"{a}/{b}: пересечение по id {rep['id_overlap']}; текст {rep['text_overlap']}; группы {rep['group_overlap']}; near-dup {rep['near_dup_pairs']}")
    passed = all(is_clean(r) for r in pairs.values())
    result = {'version': p['collect']['version'], 'threshold': nd['threshold'], 'passed': passed, 'pairs': pairs}
    Path(p['paths']['metrics_contamination']).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    if not passed:
        raise SystemExit('Обнаружена утечка между выборками')


if __name__ == '__main__':
    main()
