"""Option A (21 Nov): exponential moving average of the weights, saved as ema.pt next to last.pt."""
import torch
from torch import nn

from ps1.vision.train import make_ema


def tiny() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Flatten(), nn.LazyLinear(2))


def test_ema_off_returns_none():
    assert make_ema(tiny(), 0.0) is None


def test_first_update_copies_then_decays_and_averages_batchnorm_stats():
    model = tiny()
    model(torch.randn(2, 3, 8, 8))                          # materialise the lazy layer
    ema = make_ema(model, decay=0.9)
    ema.update_parameters(model)                            # first update: a copy
    w0 = model[0].weight.detach().clone()
    rm0 = model[1].running_mean.detach().clone()
    assert torch.allclose(ema.module[0].weight, w0)

    with torch.no_grad():
        model[0].weight.add_(1.0)
        model[1].running_mean.fill_(10.0)
    ema.update_parameters(model)
    assert torch.allclose(ema.module[0].weight, 0.9 * w0 + 0.1 * (w0 + 1.0), atol=1e-6)
    assert torch.allclose(ema.module[1].running_mean, 0.9 * rm0 + 0.1 * 10.0, atol=1e-5)   # buffers too
