from typing import Literal

from .resnet import resnet50
from .vgg16 import VGG16P2M, VGG16Recons, VGG16TensorflowAlign


def get_backbone(
    backbone: Literal["vgg16", "resnet"],
    align_with_tensorflow: bool,
    output_features_idx: list = [2, 3, 4, 5],
):
    if backbone.startswith("vgg16"):
        if align_with_tensorflow:
            nn_encoder = VGG16TensorflowAlign(output_features_idx=output_features_idx)
        else:
            nn_encoder = VGG16P2M(pretrained="pretrained" in backbone)
        nn_decoder = VGG16Recons()
    elif backbone == "resnet50":
        nn_encoder = resnet50()
        nn_decoder = None
    else:
        raise NotImplementedError(f"No implemented backbone called {backbone} found")
    return nn_encoder, nn_decoder
