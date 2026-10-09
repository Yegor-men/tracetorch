"""Shared, isolated randomness and explicit optional CUDA coverage."""

import pytest
import torch


@pytest.fixture(autouse=True)
def isolated_random_seed():
    # Test order must not affect random inputs or leak RNG changes to callers.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(0)
        yield


@pytest.fixture(params=[
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is not available")),
])
def device(request):
    return torch.device(request.param)
