"""Один интерфейс вычислений: явно синтетический стенд и настоящая модель."""
import os
from pathlib import Path
import time


class StubBackend:
    kind = 'stub'

    def __init__(self, manifest, root):
        self.loaded_id = manifest['model_id']
        self.runtime_sha256 = manifest['runtime_sha256']
        self.runtime = manifest['runtime']
        self.calls = 0

    def warmup(self):
        time.sleep(.01)

    def generate(self, prompt, limit):
        self.calls += 1
        started = time.perf_counter()
        time.sleep(.01)
        return dict(answer='STUB ONLY: ' + self.loaded_id, output_tokens=1,
                    generation_seconds=time.perf_counter() - started)


class TransformersBackend:
    kind = 'transformers'

    def __init__(self, manifest, root):
        os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.5')
        os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.4')
        from importlib.metadata import version
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.runtime import resolve_device, resolve_dtype, set_seed
        from src.deployment import safe_path
        self.torch = torch
        self.loaded_id = manifest['model_id']
        self.runtime_sha256 = manifest['runtime_sha256']
        self.runtime = manifest['runtime']
        self.calls = 0
        for package, expected in self.runtime['packages'].items():
            if version(package) != expected:
                raise ValueError('Изменилась версия библиотеки: ' + package)
        self.device = resolve_device(self.runtime['device'])
        if self.device.type == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('Нужна оценка runtime на доступном устройстве')
        set_seed(self.runtime['generation']['seed'])
        artifact = safe_path(root, manifest['artifact_dir'])
        base = safe_path(root, manifest.get('base_artifact_dir', manifest['artifact_dir']))
        self.tokenizer = AutoTokenizer.from_pretrained(str(artifact), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(base), local_files_only=True, dtype=resolve_dtype(self.runtime['dtype']))
        if manifest.get('model_kind') == 'adapter':
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(artifact), local_files_only=True)
        self.model = self.model.to(self.device).eval()

    def synchronize(self):
        if self.device.type == 'mps':
            self.torch.mps.synchronize()
        elif self.device.type == 'cuda':
            self.torch.cuda.synchronize()

    def _generate(self, prompt, limit):
        config = self.runtime['generation']
        text = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': config['system']}, {'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=config['enable_thinking'])
        batch = self.tokenizer(text, return_tensors='pt').to(self.device)
        n = batch['input_ids'].shape[1]
        if n > self.runtime['max_input_tokens']:
            raise ValueError('Слишком длинный вход после токенизации')
        self.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            result = self.model.generate(**batch, max_new_tokens=limit, do_sample=False,
                                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        self.synchronize()
        seconds = time.perf_counter() - started
        tokens = result[0, n:]
        return dict(answer=self.tokenizer.decode(tokens, skip_special_tokens=True),
                    output_tokens=len(tokens), input_tokens=n, generation_seconds=seconds)

    def warmup(self):
        self._generate('покажи список включённых будильников', self.runtime['generation']['max_new_tokens'])

    def generate(self, prompt, limit):
        self.calls += 1
        return self._generate(prompt, limit)
