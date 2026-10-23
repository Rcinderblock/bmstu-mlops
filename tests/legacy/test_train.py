"""Проверки заморозки, веса токенов и последней группы микробатчей."""
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import random
import unittest
import numpy as np
import torch
from peft import get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
import json
from src.config import load_params
from src.train import lora_config, evaluate, optimizer_batches, answer_loss
from src.runtime import set_seed
from src.compare import load_adapter_tokenizer


def example(length, value=1):
    return {'input_ids': [value]*length, 'attention_mask': [1]*length,
            'labels': [-100]+[value]*(length-1)}


class TrainTests(unittest.TestCase):
    def test_selected_logits_match_full_loss_and_gradients_with_padding(self):
        cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                       num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        model=Qwen3ForCausalLM(cfg).eval()
        batch={'input_ids':torch.tensor([[0,1,2,3,4],[1,2,3,4,5]]),
               'attention_mask':torch.tensor([[0,1,1,1,1],[1,1,1,1,1]]),
               'labels':torch.tensor([[-100,-100,-100,3,4],[-100,-100,-100,-100,5]])}
        full=model(**batch,use_cache=False).loss
        full.backward();grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        selected=answer_loss(model,batch);selected.backward()
        self.assertAlmostEqual(full.item(),selected.item(),places=6)
        for n,p in model.named_parameters():
            if n in grads:torch.testing.assert_close(p.grad,grads[n],atol=1e-6,rtol=1e-5)

    def test_freeze_restricts_real_trainable_matrices_and_preserves_backward(self):
        params=load_params()
        cfg=Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                       head_dim=8)
        sizes=[]
        for freeze in (0,2):
            model=get_peft_model(Qwen3ForCausalLM(cfg), lora_config(params,4,freeze))
            sizes.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
            for name,p in model.named_parameters():
                if p.requires_grad:
                    layer=int(name.split('layers.')[1].split('.')[0])
                    self.assertGreaterEqual(layer,freeze)
                    self.assertIn('lora_',name)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.train()
            ids=torch.tensor([[1,2,3,4]])
            model(input_ids=ids,labels=ids,use_cache=False).loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                                for p in model.parameters() if p.requires_grad))
        self.assertEqual(sizes[0],2*sizes[1])

    def test_partial_accumulation_group_is_not_lost(self):
        rows=[example(3,i+1) for i in range(9)]
        groups=list(optimizer_batches(rows,2,0,4,42))
        self.assertEqual([len(g) for g in groups],[4,1])
        self.assertEqual(sum(len(b['input_ids']) for g in groups for b in g),9)

    def test_validation_weights_by_answer_tokens_and_restores_mode(self):
        class FixedLoss(torch.nn.Module):
            def forward(self,**batch):
                return SimpleNamespace(loss=batch['input_ids'][0,-1].float())
        model=FixedLoss().eval()
        loss=evaluate(model,[example(2,2),example(4,4)],0,torch.device('cpu'),1)
        self.assertEqual(loss,3.5)  # (2*1 + 4*3)/(1+3), а не 3
        self.assertFalse(model.training)

    def test_validation_rejects_no_targets(self):
        model=torch.nn.Linear(1,1)
        with self.assertRaises(ValueError):
            evaluate(model,[example(1)],0,torch.device('cpu'),1)

    def test_seed_controls_initialization_dropout_numpy_and_python(self):
        def values():
            return (random.random(),np.random.rand(),torch.nn.Dropout(.5)(torch.ones(16)))
        set_seed(42);a=values();set_seed(42);b=values()
        self.assertEqual(a[:2],b[:2]);self.assertTrue(torch.equal(a[2],b[2]))

    def test_adapter_tokenizer_never_falls_back_to_network_or_base(self):
        with patch('src.compare.AutoTokenizer.from_pretrained') as load:
            load_adapter_tokenizer(Path('/adapter'))
            load.assert_called_once_with(Path('/adapter'),local_files_only=True)

    def test_saved_tokenizer_roundtrip_restores_training_padding_and_template(self):
        params=load_params()
        tok=AutoTokenizer.from_pretrained(params['model']['name'],local_files_only=True)
        messages=[{'role':'user','content':'поставь будильник'}]
        with TemporaryDirectory() as d:
            path=Path(d);tok.save_pretrained(path)
            (path/'training_metadata.json').write_text(json.dumps({'tokenize':{'padding_side':'left'}}))
            loaded=load_adapter_tokenizer(path)
            self.assertEqual(loaded.padding_side,'left')
            kwargs=dict(tokenize=False,add_generation_prompt=True,enable_thinking=False)
            self.assertEqual(loaded.apply_chat_template(messages,**kwargs),tok.apply_chat_template(messages,**kwargs))

    def test_invalid_layers_or_dictionary_modules_rejected(self):
        params=load_params()
        with self.assertRaises(ValueError):lora_config(params,28,28)
        params['lora']['modules_to_save']=['lm_head']
        with self.assertRaises(ValueError):lora_config(params,28,0)
