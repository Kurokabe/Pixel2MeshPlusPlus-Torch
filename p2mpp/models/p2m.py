from typing import List, Literal, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from mesh import get_base_mesh
from models.backbones import get_backbone
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection


@dataclass
class LossConfig:
    normal_weights: float = 1.0
    edge_weights: float = 1.0
    laplace_weights: float = 1.0
    move_weights: float = 1.0
    constant_weights: float = 1.0
    chamfer_weights: Tuple[float, float, float] = [1.0, 1.0, 1.0]
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
    lr_step: List[int] = [30, 45]
    lr_factor: float = 0.1


@dataclass
class BaseMeshConfig:
    name: str = "ellipsoid"
    mesh_pose: Tuple[float, float, float] = [0.0, 0.0, 0.0]


class P2MModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        coord_dim: int,
        last_hidden_dim: int,
        gconv_activation: bool,
        backbone: Literal["vgg16", "resnet"],
        align_with_tensorflow: bool,
        base_mesh_config: BaseMeshConfig,
        z_threshold: float,
        camera_f: Tuple[float, float],
        camera_c: Tuple[float, float],
        mesh_pos,
    ):
        super(P2MModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.last_hidden_dim = last_hidden_dim
        basemesh = get_base_mesh(base_mesh_config.name, base_mesh_config.mesh_pose)

        self.init_pts = nn.Parameter(basemesh.coord, requires_grad=False)
        self.gconv_activation = gconv_activation

        self.nn_encoder, self.nn_decoder = get_backbone(backbone, align_with_tensorflow)
        self.features_dim = self.nn_encoder.features_dim + self.coord_dim

        self.gcns = nn.ModuleList(
            [
                GBottleneck(
                    6,
                    self.features_dim,
                    self.hidden_dim,
                    self.coord_dim,
                    basemesh.adj_mat[0],
                    activation=self.gconv_activation,
                ),
                GBottleneck(
                    6,
                    self.features_dim + self.hidden_dim,
                    self.hidden_dim,
                    self.coord_dim,
                    basemesh.adj_mat[1],
                    activation=self.gconv_activation,
                ),
                GBottleneck(
                    6,
                    self.features_dim + self.hidden_dim,
                    self.hidden_dim,
                    self.last_hidden_dim,
                    basemesh.adj_mat[2],
                    activation=self.gconv_activation,
                ),
            ]
        )

        self.unpooling = nn.ModuleList(
            [GUnpooling(basemesh.unpool_idx[0]), GUnpooling(basemesh.unpool_idx[1])]
        )

        # if options.align_with_tensorflow:
        #     self.projection = GProjection
        # else:
        #     self.projection = GProjection
        self.projection = GProjection(
            mesh_pos,
            camera_f,
            camera_c,
            bound=z_threshold,
            tensorflow_compatible=align_with_tensorflow,
        )

        self.gconv = GConv(
            in_features=self.last_hidden_dim,
            out_features=self.coord_dim,
            adj_mat=basemesh.adj_mat[2],
        )

    def forward(self, img):
        batch_size = img.size(0)
        img_feats = self.nn_encoder(img)
        img_shape = self.projection.image_feature_shape(img)

        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)
        # GCN Block 1
        x = self.projection(img_shape, img_feats, init_pts)
        x1, x_hidden = self.gcns[0](x)

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.projection(img_shape, img_feats, x1)
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))
        # after deformation 2
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        x = self.projection(img_shape, img_feats, x2)
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        x3, _ = self.gcns[2](x)
        if self.gconv_activation:
            x3 = F.relu(x3)
        # after deformation 3
        x3 = self.gconv(x3)

        if self.nn_decoder is not None:
            reconst = self.nn_decoder(img_feats)
        else:
            reconst = None

        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [init_pts, x1_up, x2_up],
            "reconst": reconst,
        }
