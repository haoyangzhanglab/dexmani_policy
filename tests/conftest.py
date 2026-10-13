import torch


def pytest_sessionstart(session):
    # Small CPU regression fixtures; no production config or CUDA settings change.
    torch.set_num_threads(1)
