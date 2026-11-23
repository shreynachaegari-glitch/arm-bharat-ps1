import torch

from ps1.vision.dataset import MEAN, STD
from ps1.vision.export import Normalized


def test_normalisation_inside_the_deployed_graph_matches_training_preprocessing():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()).eval()
    x01 = torch.rand(2, 3, 32, 32)
    normalized = (x01 - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
    assert torch.allclose(Normalized(model)(x01), model(normalized), atol=1e-6)
