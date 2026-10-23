"""Собрать проверочный набор и доказать связь полного индекса с train.pt."""
from pathlib import Path
import hashlib
import json
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from evaluation.contract import validate_golden, write_json
from src.textnorm import normalize_text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def tokens(text):
    words = re.findall(r'\w+', normalize_text(text))
    return {tuple(words[i:i+3]) for i in range(max(1, len(words)-2))}


def main():
    source = Path('data/eval-source/annotations.json')
    annotation = json.loads(source.read_text())
    original = [json.loads(line) for line in Path('data/train.jsonl').read_text().splitlines() if line.strip()]
    blob = torch.load('data/tokenized/train.pt', weights_only=True)
    train_ids = [row['id'] for row in original]
    assert train_ids == [row['id'] for row in blob['examples']], 'Индекс обязан покрывать весь train.pt в том же порядке'
    training = [dict(id=r['id'], input=next(m['content'] for m in r['messages'] if m['role']=='user'), group_id=r['topic']) for r in original]
    labels = ['alarm_set','alarm_query','alarm_remove','calendar_set','calendar_query','calendar_remove',
              'lists_createoradd','lists_query','lists_remove','REFUSE']
    rows=[]
    for raw in annotation['golden']:
        scenario = raw['expected'].split('_')[0]
        slices=[scenario]
        if raw['expected'].endswith('_remove'):slices.append('destructive')
        rows.append({**raw, 'choices': labels, 'slices': slices})
    behavior = [{**r, 'choices': labels, 'slices':['behavioral',r['type']]} for r in annotation['behavioral']]
    proof = validate_golden(rows, training, behavior)
    train_sets = [(r,tokens(r['input'])) for r in training]
    nearest=[]
    for row in rows:
        query=tokens(row['input'])
        matches=sorted(((len(query & t)/len(query | t), r) for r,t in train_sets),key=lambda x:(-x[0],x[1]['id']))[:3]
        nearest.append(dict(id=row['id'], input=row['input'], expected=row['expected'],review=row['review'],
                            nearest=[dict(id=r['id'],input=r['input'],jaccard_3gram=round(s,6)) for s,r in matches]))
    proof.update(train_source_sha256=sha('data/train.jsonl'),train_tensor_sha256=sha('data/tokenized/train.pt'),
                 model=blob['model'],model_revision=blob['model_revision'],
                 train_source_ids_match_tensor_ids=True,annotations_sha256=sha(source),
                 golden_source_groups=len({r['group_id'] for r in rows}),
                 max_train_golden_jaccard_3gram=max(r['nearest'][0]['jaccard_3gram'] for r in nearest),
                 semantic_review='Поштучный смысловой разбор с помощью агента; человеческая проверка не заявлена. Совпадение намерения ожидаемо; отсутствие всех смысловых утечек не доказано.')
    out=Path('data/eval');out.mkdir(parents=True,exist_ok=True)
    for name,content in [('golden',rows),('behavioral',behavior),('training-index',training)]:
        write_rows(out/f'{name}.jsonl',content)
    write_json(out/'source-and-leakage.json',proof)
    write_json(out/'nearest-train-review.json',nearest)
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
