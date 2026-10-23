"""Проверки дефектов на маленьких моделях без загрузки весов."""

import unittest
from unittest.mock import patch
import torch
from src.inspect_model import (
    PeakMemory,
    device_metric_source,
    forward_hooks,
    parameter_rows,
    group_table,
    lora_params_formula,
)


class AnatomyTests(unittest.TestCase):
    def test_shared_weights_have_one_owner(self):
        model = torch.nn.Module()
        model.embed_tokens = torch.nn.Embedding(7, 3)
        model.lm_head = torch.nn.Linear(3, 7, bias=False)
        model.lm_head.weight = model.embed_tokens.weight
        rows = parameter_rows(model)
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(g["params"] for g in group_table(rows)), 21)
        self.assertEqual(rows[1]["tied_to"], "embed_tokens.weight")
        self.assertEqual(rows[1]["own_params"], 0)

    def test_hook_cleanup_after_failure_preserves_foreign_hook(self):
        layer = torch.nn.Linear(3, 4)
        existing = layer.register_forward_hook(lambda *args: None)
        with self.assertRaisesRegex(RuntimeError, "failed forward"):
            with forward_hooks({"test": layer}) as store:
                layer(torch.ones(1, 2, 3))
                self.assertEqual(len(store["test"]), 2)
                raise RuntimeError("failed forward")
        self.assertEqual(len(layer._forward_hooks), 1)
        existing.remove()

    def test_peak_keeps_transient_allocation_and_stops_thread(self):
        with (
            patch(
                "src.inspect_model.device_allocated_bytes", side_effect=[100, 900, 120]
            ),
            patch("src.inspect_model.synchronize"),
        ):
            with PeakMemory(torch.device("mps"), interval=10) as peak:
                peak.sample("temporary")
            self.assertEqual(peak.used, 900)
            self.assertEqual(peak.samples, 3)
            self.assertFalse(peak.thread.is_alive())

    def test_metric_names_follow_device(self):
        self.assertEqual(
            device_metric_source(torch.device("mps")),
            "torch.mps.driver_allocated_memory",
        )
        self.assertEqual(
            device_metric_source(torch.device("cuda")),
            "torch.cuda.max_memory_allocated",
        )

    def test_lora_uses_actual_rectangular_dimensions(self):
        model = torch.nn.Module()
        model.q_proj = torch.nn.Linear(3, 7, bias=False)
        model.v_proj = torch.nn.Linear(3, 2, bias=False)
        model.other = torch.nn.Linear(3, 9, bias=False)
        self.assertEqual(
            lora_params_formula(model, 4, ["q_proj", "v_proj"]), 4 * (3 + 7 + 3 + 2)
        )


if __name__ == "__main__":
    unittest.main()
