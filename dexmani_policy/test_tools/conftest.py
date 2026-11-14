import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(17)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield request.param
    torch.set_num_threads(previous)
