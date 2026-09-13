"""Анатомия модели: параметры по слоям, нормы активаций, профиль памяти.

    python -m src.inspect_model            полный разбор, отчёт в docs/
    python -m src.inspect_model --probe M  один режим замера памяти (служебный
                                           вызов из отдельного процесса)

Файл называется inspect_model.py, а не inspect.py: имя inspect занято
модулем стандартной библиотеки, и его перекрытие ломает импорты в чужом коде.
"""

import argparse
import gc
import json
import os
import platform
import sys
import time
import copy
import contextlib
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model

from src.config import load_params
from src.model import prepare_inputs, load_model, set_seed

# Пик RSS снимается разными механизмами на разных ОС, поэтому оба импорта
# необязательные: resource есть на macOS и Linux, но его нет на Windows;
# psutil нужен на Windows, где peak_wset — единственный high-water mark,
# который отдаёт система. Код, написанный под одну ОС, у соседа не запустится.
try:
    import resource
except ImportError:
    resource = None

try:
    import psutil
except ImportError:
    psutil = None

# Последовательная загрузка для служебных процессов измерения.
# Это настройка из заготовки; статистику чужих зависаний не выдаём за свою.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

MODES = ("inference", "full_ft", "lora")

# Порядок задаёт порядок строк в таблице. Проверка идёт сверху вниз,
# поэтому «norm» стоит после проекций: в их именах слова norm нет.
GROUPS = (
    ("embed", ("embed_tokens",)),
    ("q_proj", ("q_proj",)),
    ("k_proj", ("k_proj",)),
    ("v_proj", ("v_proj",)),
    ("o_proj", ("o_proj",)),
    ("gate_proj", ("gate_proj",)),
    ("up_proj", ("up_proj",)),
    ("down_proj", ("down_proj",)),
    ("norm", ("norm",)),
    ("lm_head", ("lm_head",)),
)


def resolve_device(params: dict) -> torch.device:
    """Развернуть device: auto в конкретное устройство — ровно один раз.

    Строка «auto» уходит в device_map и включает диспетчер accelerate,
    который для шага обучения только мешает. Решаем здесь и передаём дальше
    уже конкретное имя.
    """
    name = params["model"]["device"]
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# 1. Параметры по типам модулей
# --------------------------------------------------------------------------


def group_of(name: str) -> str:
    """Тип модуля по имени параметра."""
    for group, marks in GROUPS:
        if any(mark in name for mark in marks):
            return group
    return "прочее"


def parameter_rows(model) -> list[dict]:
    """Все тензоры параметров модели.

    remove_duplicate=False — иначе в таблицу не попадёт lm_head.
    """
    rows = []
    owners = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        owner = owners.setdefault(id(param), name)
        rows.append(
            {
                "name": name,
                "shape": tuple(param.shape),
                "numel": param.numel(),
                "tied": owner != name,
                "tied_to": owner if owner != name else None,
            }
        )
    total = sum(row["numel"] for row in rows if not row["tied"])
    for row in rows:
        row["own_params"] = 0 if row["tied"] else row["numel"]
        row["share"] = row["own_params"] / total if total else 0.0
    return rows


def group_table(rows: list[dict]) -> list[dict]:
    """Свод «тип модуля → shape → параметров → доля от всей модели».

    В params попадают только уникальные тензоры, в tied_params — то,
    что модуль переиспользует у соседа.
    """
    total = sum(r["numel"] for r in rows if not r["tied"])
    agg: dict[str, dict] = {}
    for row in rows:
        group = group_of(row["name"])
        item = agg.setdefault(
            group,
            {
                "group": group,
                "modules": 0,
                "shapes": [],
                "params": 0,
                "tied_params": 0,
            },
        )
        item["modules"] += 1
        shape = "×".join(map(str, row["shape"]))
        if shape not in item["shapes"]:
            item["shapes"].append(shape)
        if row["tied"]:
            item["tied_params"] += row["numel"]
        else:
            item["params"] += row["numel"]

    order = [g for g, _ in GROUPS] + ["прочее"]
    table = [agg[g] for g in order if g in agg]
    for item in table:
        # В группе norm форм две (по голове и по hidden), показываем обе.
        item["shape"] = ", ".join(item.pop("shapes"))
        item["share"] = item["params"] / total
    return table


# --------------------------------------------------------------------------
# 2. Forward-hooks и нормы активаций
# --------------------------------------------------------------------------


def decoder_layers(model):
    """Список декодер-блоков. У Qwen3 это model.model.layers."""
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    return decoder.layers


def hook_targets(model) -> dict[str, int]:
    """Первый, средний и последний блок — по номерам, а не по именам."""
    n_layers = len(decoder_layers(model))
    return {"первый": 0, "средний": n_layers // 2, "последний": n_layers - 1}


@contextlib.contextmanager
def forward_hooks(modules: dict):
    """Наблюдать за слоями и снять только свои хуки даже при исключении."""
    store: dict[str, list[float]] = {}
    handles = []

    def make_hook(label: str):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            store[label] = hidden[0].float().norm(dim=-1).detach().cpu().tolist()

        return hook

    try:
        for label, module in modules.items():
            handles.append(module.register_forward_hook(make_hook(label)))
        yield store
    finally:
        for handle in handles:
            handle.remove()


def activation_norms(tokenizer, model, params: dict) -> dict:
    """L2-нормы скрытых состояний на выходе трёх блоков, по позициям токена."""
    layers = decoder_layers(model)
    targets = hook_targets(model)
    inputs = prepare_inputs(tokenizer, model, params, params["hooks"]["prompt"])
    with forward_hooks({label: layers[i] for label, i in targets.items()}) as store:
        with torch.inference_mode():
            model(**inputs, use_cache=False)

    return {
        "layers": targets,
        "norms": {label: store[label] for label in targets},
        "n_tokens": inputs["input_ids"].shape[1],
    }


# --------------------------------------------------------------------------
# 3. Сколько параметров добавляет LoRA
# --------------------------------------------------------------------------


def lora_config(params: dict, cfg: dict) -> LoraConfig:
    """LoraConfig из params.yaml — ни r, ни target_modules в коде не зашиты."""
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=params["lora"]["alpha_ratio"] * cfg["r"],
        lora_dropout=params["lora"]["dropout"],
        target_modules=list(cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )


def lora_params_formula(model, r: int, target_modules) -> int:
    """Своя формула: на каждый целевой Linear ровно r * (in_features + out_features).

    A имеет форму (r, in), B — (out, r), смещений у них нет. Вся арифметика
    LoRA умещается в эту строчку, и она обязана сойтись с peft до штуки.
    """
    targets = set(target_modules)
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in targets:
            total += r * (module.in_features + module.out_features)
    return total


def lora_report(model, params: dict) -> list[dict]:
    """Для каждого конфига: своя формула против peft.

    Адаптер снимается через unload(): дальше модель нужна чистой.
    """
    base_params = sum(p.numel() for p in model.parameters())
    result = []
    for cfg in params["lora"]["configs"]:
        expected = lora_params_formula(model, cfg["r"], cfg["target_modules"])

        original_grad = {id(p): p.requires_grad for p in model.parameters()}
        peft_model = get_peft_model(model, lora_config(params, cfg))
        try:
            trainable, total = peft_model.get_nb_trainable_parameters()
        finally:
            model = peft_model.unload()
            for p in model.parameters():
                p.requires_grad_(original_grad[id(p)])

        result.append(
            {
                "name": cfg["name"],
                "r": cfg["r"],
                "target_modules": list(cfg["target_modules"]),
                "formula": expected,
                "peft": trainable,
                "match": expected == trainable,
                "total_with_adapter": total,
                "share_of_base": trainable / base_params,
            }
        )
    return result


# --------------------------------------------------------------------------
# 4. Память в трёх режимах
# --------------------------------------------------------------------------


def device_allocated_bytes(device: torch.device) -> int:
    """Сколько памяти занято прямо сейчас."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    used, _ = peak_rss()
    return used


def device_metric_source(device: torch.device) -> str:
    """Имя функции, которой снята память."""
    if device.type == "mps":
        return "torch.mps.driver_allocated_memory"
    if device.type == "cuda":
        return "torch.cuda.max_memory_allocated"
    _, source = peak_rss()
    return source


def peak_rss() -> tuple[int, str]:
    """Пик RSS процесса в байтах И метка источника метрики.

    Метка возвращается не для красоты: «пик 1001 МБ» без указания, чем это
    снято, — не результат, а повод для спора. Тем более что RSS и память
    ускорителя — разные величины (см. PeakMemory ниже).

    Три ОС меряют по-разному:

    * macOS и Linux — `resource.getrusage(RUSAGE_SELF).ru_maxrss`, high-water
      mark процесса; на macOS он в байтах, на Linux в килобайтах;
    * Windows — `psutil.Process().memory_info().peak_wset`: модуля `resource`
      там нет вовсе. Обратное тоже верно — поля `peak_wset` нет на macOS и
      Linux, и код, написанный только под него, у соседа падает.

    Если недоступно ничего — исключение. Тихий ноль хуже отсутствия числа:
    ноль попадает в отчёт и его выдают за результат.
    """
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (peak if sys.platform == "darwin" else peak * 1024), "ru_maxrss"
    if psutil is not None and hasattr(psutil.Process().memory_info(), "peak_wset"):
        return int(psutil.Process().memory_info().peak_wset), "peak_wset"
    raise RuntimeError(
        f"нечем снять пик RSS на платформе {sys.platform}: модуля resource нет, "
        "а psutil не установлен либо не отдаёт peak_wset. Выполните uv sync."
    )


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class PeakMemory:
    """Пик во время вычислений, с фоновым опросом и контрольными точками.

    MPS не предоставляет счётчик исторического пика. Поэтому максимум
    наблюдений — нижняя оценка истинного пика: короткий всплеск между
    опросами может остаться незамеченным. CUDA отдаёт собственный максимум.
    """

    def __init__(self, device: torch.device, interval: float = 0.01):
        self.device = device
        self.used = 0
        if interval <= 0:
            raise ValueError("Интервал опроса должен быть положительным")
        self.interval = interval
        self.samples = 0
        self.phases = {}
        self.peak_history = []
        self.error = None
        self.stop = threading.Event()
        self.lock = threading.Lock()

    def sample(self, phase=None):
        value = device_allocated_bytes(self.device)
        with self.lock:
            self.samples += 1
            if value > self.used:
                self.used = value
                self.peak_history.append(
                    [round(time.perf_counter() - self.started, 4), value]
                )
            if phase is not None:
                self.phases[phase] = round(value / 1024**2, 2)

    def _poll(self):
        try:
            while not self.stop.wait(self.interval):
                self.sample()
        except Exception as error:
            self.error = error

    def __enter__(self) -> "PeakMemory":
        synchronize(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        self.sample("start")
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        try:
            synchronize(self.device)
            self.sample("end")
            if self.device.type == "cuda":
                self.used = max(self.used, torch.cuda.max_memory_allocated(self.device))
        finally:
            self.stop.set()
            self.thread.join()
        if self.error is not None and exc[0] is None:
            raise RuntimeError("Ошибка опроса памяти") from self.error
        return False

    def result(self) -> dict:
        """Числа замера вместе с именем метрики, которой они сняты."""
        rss, rss_source = peak_rss()
        accelerator = self.device.type in ("mps", "cuda")
        return {
            "peak_mb": round((self.used if accelerator else rss) / 1024**2, 1),
            "peak_device_mb": round(self.used / 1024**2, 1),
            "peak_rss_mb": round(rss / 1024**2, 1),
            "metric": (
                f"аллокатор {self.device.type}" if accelerator else "RSS процесса"
            ),
            "metric_source": (
                device_metric_source(self.device) if accelerator else rss_source
            ),
            "rss_source": rss_source,
            "sample_interval_seconds": self.interval,
            "sample_count": self.samples,
            "phases_device_mb": self.phases,
            "peak_history_seconds_bytes": self.peak_history,
        }


def measure_mode(mode: str, params: dict) -> dict:
    """Один режим: инференс / full fine-tune / LoRA.

    Обучение — ровно один шаг forward + backward + optimizer.step():
    пик памяти достигается уже на нём, гонять эпоху незачем.
    """
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим {mode}")
    params = copy.deepcopy(params)
    device = resolve_device(params)
    params["model"]["device"] = str(device)
    set_seed(params["generate"]["seed"])

    started = time.perf_counter()
    loss = None

    _, model = load_model(params)
    model.config.use_cache = False
    base_weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    optimizer_bytes = gradient_bytes = 0
    trainable_params = 0
    optimizer_dtypes = []

    with PeakMemory(
        device, params["memory"].get("sample_interval_seconds", 0.01)
    ) as peak:
        ids = torch.randint(
            0,
            model.config.vocab_size,
            (params["memory"]["batch_size"], params["memory"]["seq_len"]),
            device=model.device,
        )
        if mode == "inference":
            model.eval()
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)
            synchronize(device)
            peak.sample("forward")
        else:
            if mode == "lora":
                model = get_peft_model(
                    model, lora_config(params, params["lora"]["configs"][0])
                )
            model.train()
            trainable = [p for p in model.parameters() if p.requires_grad]
            trainable_params = sum(p.numel() for p in trainable)
            optimizer = torch.optim.AdamW(
                trainable,
                lr=float(params["memory"]["lr"]),
                foreach=False,
            )
            peak.sample("adapter_and_optimizer")
            output = model(input_ids=ids, labels=ids, use_cache=False)
            synchronize(device)
            peak.sample("forward")
            output.loss.backward()
            synchronize(device)
            peak.sample("backward")
            gradient_bytes = sum(
                p.grad.numel() * p.grad.element_size()
                for p in trainable
                if p.grad is not None
            )
            optimizer.step()
            synchronize(device)
            peak.sample("optimizer_step")
            state_tensors = [
                v
                for state in optimizer.state.values()
                for v in state.values()
                if isinstance(v, torch.Tensor)
            ]
            optimizer_bytes = sum(v.numel() * v.element_size() for v in state_tensors)
            optimizer_dtypes = sorted({str(v.dtype) for v in state_tensors})
            optimizer.zero_grad(set_to_none=True)
            loss = round(output.loss.detach().item(), 4)

    result = peak.result()
    result.update(
        mode=mode,
        device=str(device),
        seq_len=params["memory"]["seq_len"],
        batch_size=params["memory"]["batch_size"],
        seconds=round(time.perf_counter() - started, 1),
        loss=loss,
        pid=os.getpid(),
        seed=params["generate"]["seed"],
        dtype=params["model"]["dtype"],
        use_cache=False,
        optimizer="AdamW, foreach=False" if mode != "inference" else None,
        trainable_params=trainable_params,
        base_weights_bytes=base_weights_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_bytes,
        optimizer_state_dtypes=optimizer_dtypes,
    )
    return result


def memory_profile(params: dict) -> list[dict]:
    """Профиль памяти в трёх режимах.

    memory.repeats задаёт число прогонов на режим; берётся худший (максимум).
    """
    repeats = max(1, int(params["memory"].get("repeats", 1)))
    results = []
    for mode in MODES:
        runs = []
        for _ in range(repeats):
            child = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.inspect_model",
                    "--probe",
                    mode,
                    "--params-stdin",
                ],
                input=json.dumps(params),
                text=True,
                capture_output=True,
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                env={**os.environ, "HF_DEACTIVATE_ASYNC_LOAD": "1"},
            )
            runs.append(json.loads(child.stdout.strip().splitlines()[-1]))
        worst = dict(max(runs, key=lambda item: item["peak_mb"]))
        worst["repeats"] = repeats
        worst["peak_mb_runs"] = [item["peak_mb"] for item in runs]
        worst["run_pids"] = [item["pid"] for item in runs]
        results.append(worst)
        gc.collect()
    return results


# --------------------------------------------------------------------------
# 5. Условия, без которых цифры замера ничего не значат
# --------------------------------------------------------------------------


def environment(params: dict, memory: list[dict]) -> dict:
    """Всё, что нужно, чтобы чужой замер можно было сравнить со своим.

    Расхождение в полтора раза между двумя машинами — норма, а не ошибка,
    но только если написано, чем эти машины отличались. Метрики памяти берутся
    из самих замеров, а не из предположений: что реально сработало в дочернем
    процессе, то и уходит в отчёт.
    """

    def unique(field: str) -> str:
        values = dict.fromkeys(str(item.get(field) or "") for item in memory)
        return ", ".join(value for value in values if value)

    return {
        "platform": platform.platform(),
        "system": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "device": params["model"]["device"],
        "dtype": params["model"]["dtype"],
        "seq_len": params["memory"]["seq_len"],
        "batch_size": params["memory"]["batch_size"],
        "repeats": max(1, int(params["memory"].get("repeats", 1))),
        "memory_metric": unique("metric_source"),
        "rss_metric": unique("rss_source"),
        "sample_interval_seconds": params["memory"].get(
            "sample_interval_seconds", 0.01
        ),
        "use_cache": False,
        "optimizer": "AdamW, foreach=False",
    }


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Разбор модели: параметры, активации, память"
    )
    parser.add_argument(
        "--probe", choices=MODES, help="служебный режим: замерить память и выйти"
    )
    parser.add_argument("--params-stdin", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    params = json.load(sys.stdin) if args.params_stdin else load_params()
    set_seed(params["generate"]["seed"])
    params["model"]["device"] = str(resolve_device(params))

    if args.probe:
        with contextlib.redirect_stdout(sys.stderr):
            result = measure_mode(args.probe, params)
        print(json.dumps(result, ensure_ascii=False))
        return

    # Импорт здесь, а не наверху: matplotlib не нужен в служебных --probe
    # процессах, а тянется он заметно дольше остального.
    from src.report import write_report

    tokenizer, model = load_model(params)
    rows = parameter_rows(model)
    table = group_table(rows)
    total = sum(item["params"] for item in table)
    report = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": params["model"]["name"],
        "dtype": params["model"]["dtype"],
        "device": params["model"]["device"],
        "config": {
            key: getattr(model.config, key)
            for key in (
                "num_hidden_layers",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "tie_word_embeddings",
            )
        },
        "params_total": total,
        "params_direct": sum(p.numel() for p in model.parameters()),
        "params_by_group": table,
        "parameter_rows": rows,
        "activations": activation_norms(tokenizer, model, params),
        "lora": lora_report(model, params),
    }
    # Родитель освобождает модель до запуска измерительных процессов.
    # Их память и состояния оптимизатора не переходят в следующий режим.
    del model, tokenizer
    gc.collect()
    if params["model"]["device"].startswith("mps"):
        torch.mps.empty_cache()
    elif params["model"]["device"].startswith("cuda"):
        torch.cuda.empty_cache()
    print(
        "Параметры, хуки и LoRA рассчитаны; измеряю память в трёх отдельных процессах.",
        flush=True,
    )
    memory = memory_profile(params)
    report["memory"] = memory
    report["environment"] = environment(params, memory)

    Path(params["report"]["json"]).parent.mkdir(exist_ok=True)
    Path(params["report"]["json"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(report, params)

    env = report["environment"]
    print(
        f"\nПараметров: {total:,} (по таблице) / {report['params_direct']:,} (напрямую)".replace(
            ",", " "
        )
    )
    print(
        f"Условия: {env['platform']}, device {env['device']}, dtype {env['dtype']}, "
        f"seq_len {env['seq_len']}, прогонов на режим {env['repeats']}, "
        f"torch {env['torch']}, transformers {env['transformers']}"
    )
    for mode in report["memory"]:
        print(
            f"  {mode['mode']:<10} пик {mode['peak_mb']:>8.1f} МБ  "
            f"({mode['metric']}: {mode['metric_source']})"
        )
    print(f"\nОтчёт: {params['report']['markdown']}, график: {params['hooks']['plot']}")


if __name__ == "__main__":
    main()
