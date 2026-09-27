"""Проверки реальных границ маски и ошибок подготовки без загрузки весов."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings
import os
import subprocess
import sys
import yaml

from transformers import AutoTokenizer

from src.collate import DynamicPaddingCollator
from src.config import load_params
from src.model import build_prompt
from src.prompt import build_chat_text, prompt_token_len
from src.tokenize_data import encode_example, process_split, truncation_stats


class TokenizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = load_params()
        cls.tok = AutoTokenizer.from_pretrained(cls.params['model']['name'])
        cls.tok.padding_side = 'left'
        cls.row = {'id': 'example', 'messages': [
            {'role': 'system', 'content': 'верни только метку действия'},
            {'role': 'user', 'content': 'поставь будильник на семь утра'},
            {'role': 'assistant', 'content': 'alarm_set'}]}

    def encode(self, limit=224):
        return encode_example(self.tok, self.row, self.params, limit)

    def test_loss_contains_answer_and_eos_but_not_thinking_or_prompt(self):
        e = self.encode()
        labels = [x for x in e['labels'] if x != -100]
        self.assertEqual(self.tok.decode(labels, skip_special_tokens=True), 'alarm_set')
        self.assertEqual(labels[-1], self.tok.eos_token_id)
        self.assertGreater(e['labels'].count(-100), 0)
        self.assertEqual(e['labels'][-1], -100)  # перевод строки после EOS
        self.assertEqual(e['attention_mask'], [1] * len(e['input_ids']))

    def test_training_prefix_and_existing_inference_use_same_template(self):
        m = self.row['messages']
        full = build_chat_text(self.tok, m, self.params)
        prefix = build_chat_text(self.tok, m, self.params, True)
        self.assertTrue(full.encode().startswith(prefix.encode()))
        self.assertEqual(prefix, build_chat_text(self.tok, m[:-1], self.params, True))
        text = m[1]['content']
        self.assertEqual(build_prompt(self.tok, self.params, text),
                         build_chat_text(self.tok, [m[1]], self.params, True))

    def test_truncation_before_answer_masks_everything(self):
        boundary = self.encode()['_meta']['prompt_len']
        e = self.encode(boundary)
        self.assertEqual(e['_meta']['supervised'], 0)
        self.assertTrue(e['_meta']['truncated'])
        self.assertTrue(all(x == -100 for x in e['labels']))

    def test_one_answer_token_retained_at_boundary(self):
        boundary = self.encode()['_meta']['prompt_len']
        e = self.encode(boundary + 1)
        self.assertEqual(e['_meta']['supervised'], 1)
        self.assertEqual(e['labels'][-1], e['input_ids'][-1])

    def test_exact_limit_is_not_truncation(self):
        n = self.encode()['_meta']['full_len']
        self.assertFalse(self.encode(n)['_meta']['truncated'])
        self.assertTrue(self.encode(n - 1)['_meta']['truncated'])

    def test_bpe_merged_token_is_masked(self):
        text = 'Привет, мир'
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        n, fallback = prompt_token_len(self.tok, 'Приве', enc['input_ids'], enc['offset_mapping'])
        self.assertTrue(fallback)
        self.assertGreaterEqual(enc['offset_mapping'][n][0], len('Приве'))

    def test_left_padding_preserves_alignment_and_ignores_pad_in_loss(self):
        a = {'input_ids': [11, 12], 'attention_mask': [1, 1], 'labels': [-100, 12]}
        b = {'input_ids': [21, 22, 23, 24], 'attention_mask': [1]*4, 'labels': [-100, -100, 23, 24]}
        batch = DynamicPaddingCollator(0)([a, b])
        self.assertEqual(list(batch['input_ids'].shape), [2, 4])
        self.assertEqual(batch['input_ids'][0].tolist(), [0, 0, 11, 12])
        self.assertEqual(batch['attention_mask'][0].tolist(), [0, 0, 1, 1])
        self.assertEqual(batch['labels'][0].tolist(), [-100, -100, -100, 12])

    def test_statistics_include_dropped_rows_and_threshold_fails(self):
        params = deepcopy(self.params); params['tokenize']['max_seq_len'] = 1
        with TemporaryDirectory() as d:
            p = Path(d) / 'rows.jsonl'; p.write_text(json.dumps(self.row) + '\n')
            with warnings.catch_warnings(record=True) as emitted:
                examples, stats = process_split(self.tok, 'test', p, params)
            self.assertEqual(examples, [])
            self.assertEqual(stats['examples_in'], 1)
            self.assertEqual(stats['dropped_no_supervision'], 1)
            self.assertEqual(stats['truncated_ratio'], 1)
            self.assertFalse(stats['truncation_pass'])
            self.assertEqual(len(emitted), 1)

    def test_empty_answer_and_empty_batch_rejected(self):
        row = deepcopy(self.row); row['messages'][-1]['content'] = ' '
        with self.assertRaises(ValueError):
            encode_example(self.tok, row, self.params, 224)
        with self.assertRaises(ValueError):
            DynamicPaddingCollator(0)([])

    def test_different_prefix_is_rejected_before_masking(self):
        class BrokenTemplate:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
                return 'one' if add_generation_prompt else 'another'
        with self.assertRaisesRegex(ValueError, 'разошлись'):
            encode_example(BrokenTemplate(), self.row, self.params, 224)

    def test_stage_rejects_excessive_truncation_and_leaves_diagnostics(self):
        params = deepcopy(self.params)
        params['tokenize']['max_seq_len'] = 1
        params['train_estimate']['generation_benchmark'] = str(
            Path(params['train_estimate']['generation_benchmark']).resolve())
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'rows.jsonl').write_text(json.dumps(self.row) + '\n')
            params['data'].update(train_jsonl='rows.jsonl', val_jsonl='rows.jsonl')
            (root / 'params.yaml').write_text(yaml.safe_dump(params))
            result = subprocess.run([sys.executable, '-m', 'src.tokenize_data'], cwd=root,
                                    env={**os.environ, 'PYTHONPATH': str(Path.cwd())},
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Проверка обрезки не пройдена', result.stderr)
            stats = json.loads((root / 'metrics/tokenize.json').read_text())
            self.assertEqual(stats['splits']['train']['truncated_ratio'], 1)
            self.assertTrue((root / 'docs/tokenize_report.md').exists())


if __name__ == '__main__':
    unittest.main()
