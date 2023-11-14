from typing import List, Literal, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mesh import get_base_mesh
from .backbones import get_backbone
from .layers.gbottleneck import GBottleneck
from .layers.gconv import GConv
from .layers.gpooling import GUnpooling
from .layers.gprojection import GProjection
from ..configs import BaseMeshConfig


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
    ):
        super(P2MModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.last_hidden_dim = last_hidden_dim
        basemesh = get_base_mesh(base_mesh_config.name, base_mesh_config.mesh_pose)

        self.init_pts = nn.Parameter(basemesh.coord, requires_grad=False)
        self.gconv_activation = gconv_activation

        self.nn_encoder, self.nn_decoder = get_backbone(backbone, align_with_tensorflow)
        self.features_dim = (
            self.nn_encoder.features_dim * 3
        ) + self.coord_dim  # x 3 because of max, mean and std

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
            base_mesh_config.mesh_pose,
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

    def forward(self, img, poses):
        batch_size = img.size(0)
        img_feats = [self.nn_encoder(img[:, i]) for i in range(img.shape[1])]
        img_shape = self.projection.image_feature_shape(img)

        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)
        # GCN Block 1
        x = self.projection(img_shape, img_feats, init_pts, poses)
        x1, x_hidden = self.gcns[0](x)

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.projection(img_shape, img_feats, x1, poses)
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))
        # after deformation 2
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        x = self.projection(img_shape, img_feats, x2, poses)
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        x3, _ = self.gcns[2](x)
        if self.gconv_activation:
            x3 = F.relu(x3)
        # after deformation 3
        x3 = self.gconv(x3)

        if self.nn_decoder is not None:
            reconst = [self.nn_decoder(img_feat) for img_feat in img_feats]
            reconst = torch.stack(reconst, dim=1)
        else:
            reconst = None

        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [init_pts, x1_up, x2_up],
            "reconst": reconst,
        }
