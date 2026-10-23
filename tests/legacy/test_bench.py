"""Проверки замеров без скачивания модели и зависимости от скорости машины."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src import bench
from src.config import load_params


class BenchmarkTests(unittest.TestCase):
    def test_timers_exclude_preparation_warmup_and_previous_runs(self):
        params = load_params()
        params["bench"].update(warmup_runs=1, measure_runs=3)
        now = [0.0]
        model = SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype="test", config=SimpleNamespace())

        def load(_):
            now[0] += 100
            return object(), model

        def prepare(*_):
            now[0] += 10
            return object()

        work = iter([(50, 1), (2, 20), (4, 80), (20, 40)])

        def generate(*_):
            duration, tokens = next(work)
            now[0] += duration
            return [0] * tokens

        with (
            patch.object(bench, "set_seed"),
            patch.object(bench, "load_model", side_effect=load),
            patch.object(bench, "prepare_inputs", side_effect=prepare),
            patch.object(bench, "generate_tokens", side_effect=generate),
            patch.object(bench.time, "perf_counter", side_effect=lambda: now[0]),
            patch.object(bench, "peak_rss_mb", return_value=123),
        ):
            result = bench.run_benchmark(params)
        self.assertEqual(result["load_time_sec"], 100)
        self.assertEqual(result["tokens_per_sec_all"], [10, 20, 2])
        self.assertEqual(result["tokens_per_sec"], 10)
        self.assertEqual([r["new_tokens"] for r in result["runs"]], [20, 80, 40])

    def test_warmup_is_required(self):
        params = load_params()
        params["bench"]["warmup_runs"] = 0
        with self.assertRaises(ValueError):
            bench.run_benchmark(params)

    def test_measurement_is_required(self):
        params = load_params()
        params["bench"]["measure_runs"] = 0
        with self.assertRaises(ValueError):
            bench.run_benchmark(params)

    def test_rss_units_on_macos_and_linux(self):
        for system, value in [("darwin", 2 * 1024**2), ("linux", 2 * 1024)]:
            with (
                self.subTest(system=system),
                patch.object(bench.sys, "platform", system),
                patch.object(bench.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=value)),
            ):
                self.assertEqual(bench.peak_rss_mb(), 2)

    def test_waits_for_the_selected_accelerator(self):
        with patch.object(bench.torch.mps, "synchronize") as mps, patch.object(
            bench.torch.cuda, "synchronize"
        ) as cuda:
            bench.synchronize(SimpleNamespace(type="cpu"))
            mps.assert_not_called()
            cuda.assert_not_called()
            bench.synchronize(SimpleNamespace(type="mps"))
            mps.assert_called_once_with()
            device = SimpleNamespace(type="cuda")
            bench.synchronize(device)
            cuda.assert_called_once_with(device)


if __name__ == "__main__":
    unittest.main()
