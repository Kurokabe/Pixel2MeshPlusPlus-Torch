from typing import Optional, Tuple

import torch
import torch.nn as nn

from p2mpp.configs.config import BaseMeshConfig
from p2mpp.models.backbones import get_backbone
from p2mpp.models.layers.deformation import SampleHypothesis
from p2mpp.models.layers.gbottleneck import DeformationReasoning
from p2mpp.models.layers.gprojection import GProjection
from p2mpp.models.mesh import get_base_mesh
from p2mpp.models.mesh.hypothesis_shape import HypothesisShape


class P2MPPModel(nn.Module):
    def __init__(
        self,
        align_with_tensorflow: bool,
        base_mesh_config: BaseMeshConfig,
        z_threshold: float,
        camera_f: Tuple[float, float],
        camera_c: Tuple[float, float],
        backbone: str,
        hypothesis_shape: HypothesisShape,
        nn_encoder_ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.hypothesis_shape = hypothesis_shape

        self.nn_encoder, _ = get_backbone(
            backbone,
            align_with_tensorflow=align_with_tensorflow,
            output_features_idx=[0, 1, 2],
        )
        if nn_encoder_ckpt_path is not None:
            self.restore_nn_encoder(nn_encoder_ckpt_path)

        hypothesis_vertices = torch.Tensor(
            self.hypothesis_shape.hypothesis_vertices
        ).cuda()  # TODO change cuda to device
        adj_mat = torch.Tensor(
            self.hypothesis_shape.adj_mat
        ).cuda()  # TODO change cuda to device

        self.sample_1 = SampleHypothesis(hypothesis_vertices)
        self.projection = GProjection(
            base_mesh_config.mesh_pose,
            camera_f,
            camera_c,
            bound=z_threshold,
            tensorflow_compatible=align_with_tensorflow,
        )
        self.drb_1 = DeformationReasoning(
            in_dim=3 + 3 * (16 + 32 + 64),
            hidden_dim=192,
            out_dim=1,
            adj_mat=adj_mat,
            sample_coord=hypothesis_vertices,
        )

    def restore_nn_encoder(self, ckpt_path: str):
        nn_encoder_ckpt = torch.load(ckpt_path)
        state_dict = {
            key.replace("model.nn_encoder.", ""): value
            for key, value in nn_encoder_ckpt["state_dict"].items()
            if key.startswith("model.nn_encoder.")
        }
        self.nn_encoder.load_state_dict(state_dict)

    def forward(self, vertices, images, poses):
        batch_size = vertices.shape[0]
        num_points = vertices.shape[1]
        img_feats = [self.nn_encoder(images[:, i]) for i in range(images.shape[1])]
        img_shape = self.projection.image_feature_shape(images)

        hypothesis_vertices = self.sample_1(vertices)
        projected_vertices = self.projection(
            img_shape, img_feats, hypothesis_vertices, poses
        )
        projected_vertices = projected_vertices.reshape(
            batch_size, num_points, self.hypothesis_shape.num_hypothesis, -1
        )
        new_vertices = self.drb_1(projected_vertices, vertices)
        return {"pred_coord": new_vertices, "pred_coord_before_deform": vertices}
