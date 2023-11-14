from typing import List, Tuple, Literal
from dataclasses import dataclass


@dataclass
class LossConfig:
    normal_weights: float = 1.0
    edge_weights: float = 1.0
    laplace_weights: float = 1.0
    move_weights: float = 1.0
    constant_weights: float = 1.0
    chamfer_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    chamfer_opposite_weights: float = 1.0
    reconst_weights: float = 1.0


@dataclass
class OptimConfig:
    name: str = "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    sgd_momentum: float = 0.9
    lr: float = 5e-5
    weight_decay: float = 1e-6
    lr_step: List[int] = (30, 45)
    lr_factor: float = 0.1


@dataclass
class BaseMeshConfig:
    name: str = "ellipsoid"
    mesh_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class NetworkConfig:
    hidden_dim: int = 192
    coord_dim: int = 3
    last_hidden_dim: int = 192
    gconv_activation: bool = True
    backbone: Literal["vgg16", "resnet"] = "vgg16"
    align_with_tensorflow: bool = True
    base_mesh_config: BaseMeshConfig = BaseMeshConfig()
    z_threshold: float = 0
    camera_f: Tuple[float, float] = (248.0, 248.0)
    camera_c: Tuple[float, float] = (111.5, 111.5)
