"""Input size follows the architecture (23 Nov, EfficientNet-B2 at 260 px), and export and runtime agree on it."""
import numpy as np
import pytest
import torch
from PIL import Image

from ps1.vision.dataset import (
    IMAGE_SIZE,
    eval_transform,
    input_size,
    resize_for,
    train_transform,
)
from ps1.vision.export import deploy_transform, sidecar_input
from ps1.vision.runtime import input_spec
from ps1.vision.train import ARCHS, build_model


def test_each_architecture_has_its_input_size():
    assert input_size("effnet_b0") == input_size("mnv3s") == IMAGE_SIZE == 224
    assert input_size("effnet_b2") == 260
    assert set(ARCHS) <= {"mnv3s", "effnet_b0", "effnet_b2"} and "effnet_b2" in ARCHS
    with pytest.raises(ValueError):
        input_size("resnet50")


def test_resize_keeps_the_224_over_256_crop_ratio():
    assert resize_for(224) == 256           # unchanged for every existing model
    assert resize_for(260) == 297


@pytest.mark.parametrize("size", [224, 260])
def test_transforms_produce_the_requested_size(size):
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8))
    for tf in (eval_transform(size), train_transform(size), deploy_transform(size)):
        assert tuple(tf(image).shape) == (3, size, size)


def test_default_transforms_are_unchanged():
    image = Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8))
    assert tuple(eval_transform()(image).shape) == (3, 224, 224)


def test_effnet_b2_builds_and_outputs_one_logit_per_class():
    model = build_model(5, pretrained=False, arch="effnet_b2").eval()
    with torch.no_grad():
        assert tuple(model(torch.rand(1, 3, 260, 260)).shape) == (1, 5)


@pytest.mark.parametrize("size", [224, 260])
def test_runtime_reads_the_size_export_writes(size):
    assert input_spec(sidecar_input(size)) == (size, resize_for(size))


def test_runtime_accepts_old_224_sidecars_without_a_resize_key():
    old = {"layout": "NCHW", "shape": [1, 3, 224, 224], "dtype": "float32"}
    assert input_spec(old) == (224, 256)


@pytest.mark.parametrize("shape", [[1, 3, 224, 260], [1, 1, 224, 224], [2, 3, 224, 224], [1, 3, 512, 512]])
def test_runtime_rejects_unexpected_shapes(shape):
    with pytest.raises(ValueError):
        input_spec({"shape": shape})
