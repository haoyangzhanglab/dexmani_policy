"""Exercise the production GPU benchmark loop with native CPU scalar losses.

CUDA timing/transfer and model/data construction are controlled substitutes;
these tests do not claim CUDA overlap or full model integration.
"""

import json
from contextlib import nullcontext
from types import SimpleNamespace as NS

import benchmark_dataset_streaming as benchmark
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    "bad_index,bad_value",
    [
        (None, 0.0),
        (0, float("nan")),
        (2, float("nan")),
        (4, float("nan")),
        (4, float("inf")),
        (4, -float("inf")),
    ],
)
def test_finite_loss_covers_warmup_and_all_measured_batches(
    tmp_path, monkeypatch, capsys, bad_index, bad_value
):
    values = [1.0] * 6  # Three warmup batches followed by three measured batches.
    if bad_index is not None:
        values[bad_index] = bad_value
    state = NS(steps=0, syncs=[], checks=[], timing_finished=False, sync_complete=False)
    config = tmp_path / "benchmark.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "dataset": {"kind": "dataset", "zarr_path": "unused"},
                "agent": {"kind": "model"},
                "action_key": "action",
                "training": {"use_bfloat16": False},
            }
        ),
        config,
    )

    class Dataset(list):
        replay_buffer = NS(episode_ends=np.array([6]))

    dataset = Dataset([{} for _ in values])

    class Model:
        def initialize_training(self):
            pass

        def load_normalizer_from_dataset(self, normalizer):
            pass

        def set_normalization_spec(self, spec):
            pass

        def cuda(self):
            return self

        def train(self):
            return self

        def zero_grad(self, **kwargs):
            pass

        def compute_loss(self, batch):
            value = values[state.steps]
            state.steps += 1
            return torch.tensor(value, requires_grad=True) * 2, {}

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    def synchronize():
        state.syncs.append(state.steps)
        if state.steps == len(values):
            state.sync_complete = True

    real_counter = benchmark.time.perf_counter

    def counter():
        if state.sync_complete:
            state.timing_finished = True
        return real_counter()

    original_isfinite = torch.isfinite

    def isfinite(value):
        assert state.timing_finished, (
            "finite check must be outside the measured interval"
        )
        state.checks.append((tuple(value.shape), value.requires_grad))
        return original_isfinite(value)

    original_stack = torch.stack

    def stack(tensors, *args, **kwargs):
        assert state.timing_finished
        assert len(tensors) == len(values)
        assert all(
            t.shape == () and not t.requires_grad and t.grad_fn is None for t in tensors
        )
        return original_stack(tensors, *args, **kwargs)

    def host_sync_forbidden(*args, **kwargs):
        pytest.fail("unexpected per-batch host transfer or scalar extraction")

    original_bool = torch.Tensor.__bool__

    def tensor_bool(tensor):
        assert state.steps == 0 or state.timing_finished, (
            "per-batch host scalar conversion"
        )
        return original_bool(tensor)

    monkeypatch.setattr(
        benchmark.hydra.utils,
        "instantiate",
        lambda cfg: dataset if cfg.kind == "dataset" else Model(),
    )
    monkeypatch.setattr(benchmark, "resolve_normalization_spec", lambda cfg: {})
    monkeypatch.setattr(
        benchmark,
        "build_normalizer",
        lambda *a: {"action": NS(params_dict={"scale": torch.ones(1)})},
    )
    monkeypatch.setattr(benchmark, "DataLoader", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(benchmark.time, "perf_counter", counter)
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "controlled CPU test")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch, "isfinite", isfinite)
    monkeypatch.setattr(torch, "stack", stack)
    monkeypatch.setattr(torch.Tensor, "item", host_sync_forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", host_sync_forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", tensor_bool)
    benchmark.measure_gpu_data_wait(
        NS(
            order="random",
            gpu_config=str(config),
            path=str(tmp_path / "unused"),
            seed=42,
            batch_size=1,
            workers=0,
            batches=3,
        )
    )
    report = json.loads(capsys.readouterr().out)
    assert report["finite_loss"] is (bad_index is None)
    assert report["warmup_batches"] == report["measured_batches"] == 3
    assert state.syncs == [3, 6]
    assert state.checks == [((6,), False)]
