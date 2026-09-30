"""Стадия train: LoRA-дообучение на выходе стадии tokenize.

    python -m src.train --variant all_layers          # полный прогон варианта
    python -m src.train --variant all_layers --max-steps 4 --out /tmp/x   # smoke

Цикл обучения написан руками, а не через Trainer: так видно всё, что
обычно прячется, — где считается val loss, как копятся градиенты, что
сохраняется рядом с адаптером.

Что сохраняется в models/adapter_<variant>/: адаптер.
В metrics/train_<variant>.json — кривые train/val loss, время, пиковая память,
число обучаемых параметров, вес адаптера и отпечаток входов.
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

# Потолок памяти Metal — до импорта torch. Без него mps занимает сколько дадут,
# и на ноутбуке с 16–32 ГБ система уходит в своп вместо внятной ошибки.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")   # нижний порог не выше верхнего

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.config import load_params
from src.data import LABEL_PAD_ID, batches, load_split
from src.runtime import allocated_bytes, memory_metric, resolve_device, resolve_dtype, set_seed, synchronize


TRAIN_CODE = ("src/train.py", "src/data.py", "src/runtime.py", "src/config.py", "src/collate.py", "uv.lock")
TRAIN_PARAMS = ("model", "data", "lora", "train", "variants")


def inputs_fingerprint(params: dict) -> str:
    """Отпечаток кода обучения и секций конфига, от которых зависит адаптер.

    check.sh сверяет его с текущим: правили train.py или lr, а адаптер
    остался от прошлого прогона — проверять его бессмысленно.
    """
    h = hashlib.sha256()
    for name in TRAIN_CODE:
        h.update(name.encode())
        h.update(Path(name).read_bytes())
    h.update(json.dumps({k: params.get(k) for k in TRAIN_PARAMS}, sort_keys=True).encode())
    for key in ("train", "val"):
        h.update(Path(params["data"][key]).read_bytes())
    return h.hexdigest()[:12]


def lora_config(params: dict, n_layers: int, freeze_first: int) -> LoraConfig:
    cfg = params["lora"]
    if not 0 <= freeze_first < n_layers:
        raise ValueError(f"freeze_first должен быть от 0 до {n_layers - 1}")
    if cfg.get("modules_to_save") or any(x in ("embed_tokens", "lm_head") for x in cfg["target_modules"]):
        raise ValueError("адаптер должен содержать только низкоранговые матрицы проекций")
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        modules_to_save=cfg.get("modules_to_save"),
        layers_to_transform=list(range(freeze_first, n_layers)),
        task_type="CAUSAL_LM",
    )


@torch.no_grad()
def evaluate(model, examples, pad_id, device, batch_size: int, memory_peaks=None) -> float:
    """Средний лосс на токен по всему val-сплиту.

    Среднее по батчам нельзя: в батчах разное число токенов под маской.
    Поэтому сумма лоссов, взвешенная числом токенов, делённая на их сумму.
    """
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch in batches(examples, batch_size, pad_id, shuffle=False, seed=0):
        batch = {k: v.to(device) for k, v in batch.items()}
        n = int((batch["labels"][:, 1:] != LABEL_PAD_ID).sum())
        if n == 0:
            continue
        loss = answer_loss(model, batch)
        if memory_peaks is not None:
            memory_peaks.append(allocated_bytes(device))
        total += loss.item() * n
        count += n
    model.train(was_training)
    if device.type == "mps":
        torch.mps.empty_cache()   # логиты оценки не должны висеть в кэше до конца обучения
    if not count:
        raise ValueError("в проверочных данных нет токенов ответа")
    return total / count


def answer_loss(model, batch):
    """Читаем весь контекст; словарь считаем лишь перед целевыми токенами.

    Явно передаём сдвинутые цели библиотеке, чтобы она не сдвигала их снова.
    В файлах labels по-прежнему не сдвинуты. Удаляем только позиции, в которых
    у ВСЕХ строк цель -100; сумма, знаменатель и градиенты ошибки сохраняются.
    """
    shifted = torch.nn.functional.pad(batch["labels"], (0, 1), value=LABEL_PAD_ID)[:, 1:]
    positions = (shifted.detach().cpu() != LABEL_PAD_ID).any(dim=0).nonzero().flatten().to(shifted.device)
    if not len(positions):
        raise ValueError("нет токенов ответа")
    return model(**batch, logits_to_keep=positions,
                 shift_labels=shifted[:, positions].contiguous(), use_cache=False).loss


def optimizer_batches(examples, batch_size, pad_id, grad_accum, seed):
    """Группы микробатчей; последняя неполная группа тоже обновляет веса."""
    group = []
    for batch in batches(examples, batch_size, pad_id, shuffle=True, seed=seed):
        group.append(batch)
        if len(group) == grad_accum:
            yield group
            group = []
    if group:
        yield group


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all_layers")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None, help="куда писать адаптер и метрики (smoke-тесты)")
    ap.add_argument("--val-limit", type=int, default=None, help="оценивать на первых N примерах val (smoke)")
    args = ap.parse_args()

    params = load_params()
    variants = {v["name"]: v for v in params["variants"]}
    if args.variant not in variants:
        raise SystemExit(f"нет варианта {args.variant!r}, есть: {sorted(variants)}")
    variant = variants[args.variant]
    tcfg = params["train"]
    max_steps = args.max_steps if args.max_steps is not None else tcfg.get("max_steps")

    device = resolve_device(params["model"]["device"])
    dtype = resolve_dtype(params["model"]["dtype"])
    set_seed(tcfg["seed"])
    if any(tcfg[k] <= 0 for k in ("epochs", "batch_size", "grad_accum", "eval_every", "eval_batch_size")):
        raise ValueError("размеры батчей, эпохи и интервалы должны быть положительными")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps должен быть положительным")

    train_blob = load_split(params["data"]["train"])
    val_blob = load_split(params["data"]["val"])
    if args.val_limit:
        val_blob["examples"] = val_blob["examples"][:args.val_limit]
    pad_id = train_blob["pad_token_id"]
    if val_blob["pad_token_id"] != pad_id or any(b["model"] != params["model"]["name"] for b in (train_blob, val_blob)):
        raise ValueError("модель и токен заполнения должны совпадать с подготовленными данными")

    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(params["model"]["name"], dtype=dtype).to(device)
    n_layers = model.config.num_hidden_layers
    if params["train"].get("gradient_checkpointing"):
        # Активации 28 слоёв не храним, а пересчитываем на обратном проходе:
        # памяти в разы меньше, шаг примерно на треть дольше.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, lora_config(params, n_layers, variant["freeze_first"]))
    # Нерекурсивное пересчитывание активаций не требует градиента входных
    # эмбеддингов. Иначе даже полностью замороженные нижние слои строят граф.
    if hasattr(model, "disable_input_require_grads") and hasattr(model, "_require_grads_hook"):
        model.disable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    model.train()

    examples = train_blob["examples"]
    micro_per_epoch = math.ceil(len(examples) / tcfg["batch_size"])
    steps_per_epoch = math.ceil(micro_per_epoch / tcfg["grad_accum"])
    total_steps = steps_per_epoch * tcfg["epochs"]
    if max_steps:
        total_steps = min(total_steps, max_steps)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_steps * tcfg["warmup_ratio"])), total_steps
    )

    eval_bs = tcfg.get("eval_batch_size", tcfg["batch_size"])
    print(f"[{args.variant}] устройство {device}, обучаемых {trainable:,} из {total:,} "
          f"({trainable / total:.3%}); шагов {total_steps}", flush=True)
    synchronize(device)
    initial_started = time.perf_counter()
    memory_peaks = [allocated_bytes(device)]
    base_val = evaluate(model, val_blob["examples"], pad_id, device, eval_bs, memory_peaks)
    synchronize(device)
    initial_eval_seconds = time.perf_counter() - initial_started
    curve_train: list[list[float]] = []
    curve_val: list[list[float]] = [[0, round(base_val, 6)]]
    print(f"  шаг 0: val {base_val:.6f} ({initial_eval_seconds:.1f} с)", flush=True)

    peak = max(memory_peaks)
    started = time.perf_counter()
    step, diverged, tokens = 0, False, 0
    eval_seconds = 0.0
    for epoch in range(tcfg["epochs"]):
        for group in optimizer_batches(examples, tcfg["batch_size"], pad_id, tcfg["grad_accum"], tcfg["seed"] + epoch):
            counts = [int((b["labels"][:, 1:] != LABEL_PAD_ID).sum()) for b in group]
            if not sum(counts):
                raise ValueError("в группе микробатчей нет токенов ответа")
            accum_loss = 0.0
            for batch, n in zip(group, counts):
                if not n:
                    continue
                tokens += int(batch["attention_mask"].sum())
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = answer_loss(model, batch) * (n / sum(counts))
                if not torch.isfinite(loss):
                    diverged = True
                    break
                loss.backward()
                accum_loss += loss.item()
                peak = max(peak, allocated_bytes(device))
            if diverged:
                break
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            curve_train.append([step, round(accum_loss, 4)])
            if not math.isfinite(accum_loss):
                diverged = True
                print(f"  шаг {step}: лосс {accum_loss} — обучение разошлось, останавливаюсь")
                break
            if step % tcfg["eval_every"] == 0 or step == total_steps:
                synchronize(device)
                eval_started = time.perf_counter()
                val = evaluate(model, val_blob["examples"], pad_id, device, eval_bs, memory_peaks)
                synchronize(device)
                eval_seconds += time.perf_counter() - eval_started
                curve_val.append([step, round(val, 6)])
                peak = max(peak, max(memory_peaks))
                print(f"  шаг {step}/{total_steps}: train {curve_train[-1][1]:.4f}, val {val:.6f}; "
                      f"прошло {time.perf_counter()-started:.0f} с", flush=True)
            elif step == 1:
                print(f"  шаг 1/{total_steps}: train {curve_train[-1][1]:.4f}; "
                      f"{time.perf_counter()-started:.1f} с", flush=True)
            if step >= total_steps:
                break
        if diverged or step >= total_steps:
            break
    synchronize(device)
    seconds = time.perf_counter() - started - eval_seconds   # чистое обучение, без замеров val

    out_root = Path(args.out) if args.out else Path(params["paths"]["models"])
    adapter_dir = out_root / f"adapter_{args.variant}"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "training_metadata.json").write_text(json.dumps({
        "model": params["model"], "tokenize": params["tokenize"],
        "inputs_fingerprint": inputs_fingerprint(params),
        "base_revision": getattr(model.config, "_commit_hash", None),
    }, ensure_ascii=False, indent=2) + "\n")

    metrics = {
        "variant": args.variant,
        "freeze_first": variant["freeze_first"],
        "model": params["model"]["name"],
        "device": device.type,
        "dtype": params["model"]["dtype"],
        "seed": tcfg["seed"],
        "lr": tcfg["lr"],
        "effective_batch": tcfg["batch_size"] * tcfg["grad_accum"],
        "steps": step,
        "train_examples": len(examples),
        "val_examples": len(val_blob["examples"]),
        "train_tokens_processed": tokens,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_share": round(trainable / total, 6),
        "base_val_loss": base_val,
        "final_val_loss": curve_val[-1][1] if curve_val else None,
        "diverged": diverged,
        "curve_train": curve_train,
        "curve_val": curve_val,
        "seconds": round(seconds, 1),
        "eval_seconds": round(eval_seconds, 1),
        "initial_eval_seconds": round(initial_eval_seconds, 1),
        "wall_seconds": round(seconds + eval_seconds + initial_eval_seconds, 1),
        "seconds_per_step": round(seconds / max(step, 1), 3),
        "train_tokens_per_sec": round(tokens / seconds, 1) if seconds else 0,
        "peak_memory_mb": round(peak / 1048576, 1),
        "memory_metric": memory_metric(device),
        "adapter_dir": str(adapter_dir),
        "adapter_size_mb": dir_size_mb(adapter_dir),
        "inputs_fingerprint": inputs_fingerprint(params),
    }
    mdir = out_root / "metrics" if args.out else Path(params["paths"]["metrics"])
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"train_{args.variant}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{args.variant}] {step} шагов за {seconds:.0f} с; "
          f"пик памяти {metrics['peak_memory_mb']:.0f} МБ; адаптер {metrics['adapter_size_mb']} МБ -> {adapter_dir}")


if __name__ == "__main__":
    main()
