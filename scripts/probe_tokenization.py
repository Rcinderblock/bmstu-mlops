"""Проверить слишком короткий лимит без изменения основной конфигурации и данных."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from src.config import load_params
from src.tokenize_data import process_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seq-len', type=int, default=160)
    parser.add_argument('--output', default='docs/results/hw04-short-limit.json')
    args = parser.parse_args()
    params = load_params()
    params['tokenize']['max_seq_len'] = args.max_seq_len
    tok = AutoTokenizer.from_pretrained(params['model']['name'])
    tok.padding_side = params['tokenize']['padding_side']
    splits = {}
    for name in ('train', 'val'):
        _, splits[name] = process_split(tok, name, Path(params['data'][name + '_jsonl']), params)
    result = {'max_seq_len': args.max_seq_len, 'threshold': params['tokenize']['truncated_warn_ratio'],
              'note': 'Диагностический прогон. Основные params.yaml и data/tokenized не менялись.', 'splits': splits}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Диагностика:', args.output)


if __name__ == '__main__':
    main()
