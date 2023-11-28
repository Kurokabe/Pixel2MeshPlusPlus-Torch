from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2mpp.configs import LossConfig
from p2mpp.models.layers.chamfer_wrapper import ChamferDist

from ..mesh.ellipsoid import Ellipsoid


class P2MPPLoss(nn.Module):
    def __init__(self, loss_config: LossConfig, ellipsoid):
        super().__init__()
        self.normal_weights = loss_config.normal_weights
        self.edge_weights = loss_config.edge_weights
        self.laplace_weights = loss_config.laplace_weights
        self.move_weights = loss_config.move_weights
        self.constant_weights = loss_config.constant_weights
        self.chamfer_weights = loss_config.chamfer_weights
        self.chamfer_opposite_weights = loss_config.chamfer_opposite_weights

        self.l1_loss = nn.L1Loss(reduction="mean")
        self.l2_loss = nn.MSELoss(reduction="mean")
        self.chamfer_dist = ChamferDist()
        self.ellisoid = ellipsoid

        self.laplace_idx = nn.ParameterList(
            [nn.Parameter(idx, requires_grad=False) for idx in ellipsoid.laplace_idx]
        )
        self.edges = nn.ParameterList(
            [nn.Parameter(edges, requires_grad=False) for edges in ellipsoid.edges]
        )

    def edge_regularization(self, pred, edges):
        """
        :param pred: batch_size * num_points * 3
        :param edges: num_edges * 2
        :return:
        """
        return self.l2_loss(pred[:, edges[:, 0]], pred[:, edges[:, 1]]) * pred.size(-1)

    @staticmethod
    def laplace_coord(inputs, lap_idx):
        """
        :param inputs: nodes Tensor, size (n_pts, n_features = 3)
        :param lap_idx: laplace index matrix Tensor, size (n_pts, 10)
        for each vertex, the laplace vector shows: [neighbor_index * 8, self_index, neighbor_count]

        :returns
        The laplacian coordinates of input with respect to edges as in lap_idx
        """

        indices = lap_idx[:, :-2]
        invalid_mask = indices < 0
        all_valid_indices = indices.clone()
        all_valid_indices[invalid_mask] = 0  # do this to avoid negative indices

        vertices = inputs[:, all_valid_indices]
        vertices[:, invalid_mask] = 0
        neighbor_sum = torch.sum(vertices, 2)
        neighbor_count = lap_idx[:, -1].float()
        laplace = inputs - neighbor_sum / neighbor_count[None, :, None]

        return laplace

    def laplace_regularization(self, input1, input2, block_idx):
        """
        :param input1: vertices tensor before deformation
        :param input2: vertices after the deformation
        :param block_idx: idx to select laplace index matrix tensor
        :return:

        if different than 1 then adds a move loss as in the original TF code
        """

        lap1 = self.laplace_coord(input1, self.laplace_idx[block_idx])
        lap2 = self.laplace_coord(input2, self.laplace_idx[block_idx])
        laplace_loss = self.l2_loss(lap1, lap2) * lap1.size(-1)
        move_loss = (
            self.l2_loss(input1, input2) * input1.size(-1) if block_idx > 0 else 0
        )
        return laplace_loss, move_loss

    def normal_loss(self, gt_normal, indices, pred_points, adj_list):
        edges = F.normalize(
            pred_points[:, adj_list[:, 0]] - pred_points[:, adj_list[:, 1]], dim=2
        )
        nearest_normals = torch.stack([t[i] for t, i in zip(gt_normal, indices.long())])
        normals = F.normalize(nearest_normals[:, adj_list[:, 0]], dim=2)
        cosine = torch.abs(torch.sum(edges * normals, 2))
        return torch.mean(cosine)

    def image_loss(self, gt_img, pred_img):
        rect_loss = F.binary_cross_entropy(pred_img, gt_img)
        return rect_loss

    def forward(self, outputs, targets):
        """
        :param outputs: outputs from P2MModel
        :param targets: targets from input
        :return: loss, loss_summary (dict)
        """
        # import ipdb;ipdb.set_trace()
        chamfer_loss, edge_loss, normal_loss, lap_loss, move_loss = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        lap_const = [
            0.2,
            1.0,
            1.0,
        ]
        gt_coord, gt_normal = targets["points"], targets["normals"]
        pred_coord, pred_coord_before_deform = (
            outputs["pred_coord"],
            outputs["pred_coord_before_deform"],
        )

        pred_pts = pred_coord

        dist1_v, dist2_v, idx1_v, idx2_v = self.chamfer_dist(gt_coord, pred_coord)
        dist1, dist2, idx1, idx2 = self.chamfer_dist(gt_coord, pred_pts)

        chamfer_loss += self.chamfer_weights[2] * (
            torch.mean(dist1_v) + self.chamfer_opposite_weights * torch.mean(dist2_v)
        )
        chamfer_loss += self.chamfer_weights[2] * (
            torch.mean(dist1) + torch.mean(dist2)
        )
        normal_loss += self.normal_loss(gt_normal, idx2_v, pred_coord, self.edges[0])
        edge_loss += self.edge_regularization(pred_coord, self.edges[0])
        # lap, move = self.laplace_regularization(pred_coord_before_deform, pred_coord, 2)
        # lap_loss += lap_const[2] * lap

        loss = (
            chamfer_loss
            + self.normal_weights * normal_loss
            + self.edge_weights * edge_loss
            # + self.laplace_weights * lap_loss
        )

        loss = loss * self.constant_weights

        return loss, {
            "loss": loss,
            "loss_chamfer": chamfer_loss,
            # "loss_edge": edge_loss,
            "loss_laplace": lap_loss,
            # "loss_move": move_loss,
            "loss_normal": normal_loss,
        }
