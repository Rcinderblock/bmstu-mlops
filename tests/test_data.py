import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.collect import template
from src.dedup import SimilarityIndex, cross_near_duplicates, near_duplicates
from src.diversity import measure, violations
from src.schema import Example, SchemaError, iter_examples, dump
from src.split import group_label
from src.textnorm import normalize_group, shingles, normalize_text
from src.contamination import report, is_clean
from src.pii import scrub


def example(i, text='поставь будильник на завтра', group='template-a', answer='alarm_set'):
    return Example.model_validate({'id': str(i), 'topic': group, 'messages': [
        {'role':'system','content':'выбери действие'}, {'role':'user','content':text},
        {'role':'assistant','content':answer}]})


class DataTests(unittest.TestCase):
    def test_slot_values_share_group_but_different_actions_do_not(self):
        self.assertEqual(template('разбуди в [time : пять]'), template('разбуди в [time : девять]'))
        self.assertNotEqual(template('поставь [event_name : встречу]'), template('удали [event_name : встречу]'))

    def test_group_assignment_stable_and_aliases(self):
        ratios = {'train': .8, 'val': .1, 'test': .1}
        self.assertEqual(group_label(' ТЕМА — 2025 | алиас', ratios, 42), group_label('тема - 2025', ratios, 42))
        before = {g: group_label(g,ratios,42) for g in ['a','b','c']}
        after = {g: group_label(g,ratios,42) for g in ['d','a','c','e','b']}
        self.assertEqual(before, {g:after[g] for g in before})
        with self.assertRaises(ValueError): group_label('a', {'train':.7,'val':.1,'test':.1},42)

    def test_exact_jaccard_rejects_false_lsh_positive(self):
        index=SimilarityIndex(2,64,.85)
        index.insert(0,'поставь будильник на завтра')
        with patch.object(index.lsh,'query',return_value=['0']):
            self.assertEqual(index.query('удали список покупок полностью'),[])

    def test_index_catches_pairs_when_lsh_misses(self):
        index=SimilarityIndex(2,64,.85)
        index.insert(0,'поставь будильник на завтра')
        with patch.object(index.lsh,'query',return_value=[]):
            self.assertEqual(index.query('поставь будильник на завтра!'),[0])

    def test_matches_exhaustive_jaccard(self):
        texts=[' '.join(x) for x in itertools.combinations('abcdefgh',4)]
        pairs=set(cross_near_duplicates(texts,texts,1,64,.6))
        ss=[shingles(t,1) for t in texts]
        expected={(i,j) for i,a in enumerate(ss) for j,b in enumerate(ss) if len(a&b)/len(a|b)>=.6}
        self.assertEqual(pairs,expected)

    def test_near_dedup_leaves_no_lexical_overlap(self):
        texts=['напомни о встрече завтра в пять','напомни о встрече завтра в пять!', 'удали список покупок']
        self.assertEqual(near_duplicates(texts,4,64,.85),[1])

    def test_contamination_detects_all_four_types(self):
        rep=report([example(1)],[example(1)],4,64,.85)
        self.assertFalse(is_clean(rep))
        self.assertEqual([rep[k] for k in ['id_overlap','text_overlap','group_overlap','near_dup_pairs']],[1,1,1,1])
        self.assertTrue(is_clean(report([example(1)],[example(2,'удали список покупок','template-b')],4,64,.85)))

    def test_schema_reports_line_and_rejects_empty_topic(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'raw.jsonl';p.write_text(dump(example(1))+'\n{}\n')
            with self.assertRaisesRegex(SchemaError,'raw.jsonl:2'):
                list(iter_examples(p))
        with self.assertRaises(ValueError): example(1,group=' ')

    def test_pii_preserves_event_dates(self):
        text,hits=scrub('встреча 12.03.2027 почта test@example.org дата рождения 12.03.1995 +7 (999) 123-45-67')
        self.assertIn('12.03.2027',text)
        self.assertEqual(sum(hits.values()),3)
        self.assertNotIn('test@example.org',text)

    def test_diversity_flags_degenerate_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.jsonl';p.write_text(''.join(dump(example(i))+ '\n' for i in range(5)))
            from src.config import load_params
            failures=violations(measure(p,'topic'),load_params()['diversity'])
            self.assertTrue(any('системных промптов' in x for x in failures))
            self.assertTrue(any('групп ' in x for x in failures))
            self.assertTrue(any('unique_user_share' in x for x in failures))


if __name__=='__main__': unittest.main()
