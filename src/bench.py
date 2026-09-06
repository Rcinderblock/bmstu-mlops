"""Замер производительности машины на выбранной модели.

Три числа меряются РАЗДЕЛЬНО — смешивать их бессмысленно:
  * время загрузки модели  — разовая стоимость старта;
  * tokens/sec             — скорость генерации, только после прогрева;
  * пиковая RSS            — максимум за процесс, а не снимок в конце.
"""

import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import torch
import transformers

from src.config import load_params
from src.model import generate_tokens, load_model, prepare_inputs, set_seed


def peak_rss_mb() -> float:
    """Пиковая резидентная память процесса.

    ru_maxrss на macOS в байтах, на Linux в килобайтах.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 2) if sys.platform == "darwin" else peak / 1024


def synchronize(device) -> None:
    """Дождаться ускорителя: его вычисления выполняются асинхронно."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(params: dict) -> dict:
    """Раздельно измерить загрузку, генерацию после прогрева и пик RSS."""
    warmup_runs = params["bench"]["warmup_runs"]
    measure_runs = params["bench"]["measure_runs"]
    if warmup_runs < 1 or measure_runs < 1:
        raise ValueError("Нужны хотя бы один прогрев и одно измерение")
    set_seed(params["generate"]["seed"])
    prompt = params["bench"]["prompt"]

    t0 = time.perf_counter()
    tokenizer, model = load_model(params)
    synchronize(model.device)
    load_time = time.perf_counter() - t0

    inputs = prepare_inputs(tokenizer, model, params, prompt)
    for _ in range(warmup_runs):
        generate_tokens(model, params, inputs)
    synchronize(model.device)

    speeds = []
    runs = []
    for _ in range(measure_runs):
        synchronize(model.device)
        t0 = time.perf_counter()
        new_tokens = generate_tokens(model, params, inputs)
        synchronize(model.device)
        elapsed = time.perf_counter() - t0
        n_tokens = len(new_tokens)
        speeds.append(n_tokens / elapsed)
        runs.append({"seconds": elapsed, "new_tokens": n_tokens})

    # Медиана устойчивее среднего к одиночному выбросу.
    return {
        "model": params["model"]["name"],
        "device": str(model.device),
        "dtype": str(model.dtype),
        "load_time_sec": round(load_time, 6),
        "tokens_per_sec": round(statistics.median(speeds), 6),
        "tokens_per_sec_all": [round(s, 6) for s in speeds],
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "runs": runs,
        "prompt": prompt,
        "max_new_tokens": params["generate"]["max_new_tokens"],
        "seed": params["generate"]["seed"],
        "temperature": params["generate"]["temperature"],
        "enable_thinking": params["generate"]["enable_thinking"],
        "model_revision": getattr(model.config, "_commit_hash", None),
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def main() -> None:
    report = run_benchmark(load_params())
    Path("docs").mkdir(exist_ok=True)
    Path("docs/bench.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_keys = (
        "model", "device", "dtype", "load_time_sec", "tokens_per_sec",
        "peak_rss_mb", "warmup_runs", "measure_runs",
    )
    print(json.dumps({key: report[key] for key in summary_keys}, ensure_ascii=False, indent=2))
    print("Подробности каждого прогона: docs/bench.json")


if __name__ == "__main__":
    main()
