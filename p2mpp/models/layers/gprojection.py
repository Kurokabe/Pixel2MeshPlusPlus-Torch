import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Threshold

from p2mpp.utils.camera import batch_camera_trans, batch_camera_trans_inv


class GProjection(nn.Module):
    """
    Graph Projection layer, which pool 2D features to mesh

    The layer projects a vertex of the mesh to the 2D image and use
    bi-linear interpolation to get the corresponding feature.
    """

    def __init__(
        self, mesh_pos, camera_f, camera_c, bound=0, tensorflow_compatible=False
    ):
        super(GProjection, self).__init__()
        self.mesh_pos, self.camera_f, self.camera_c = mesh_pos, camera_f, camera_c
        self.threshold = None
        self.bound = 0
        self.tensorflow_compatible = tensorflow_compatible
        if self.bound != 0:
            self.threshold = Threshold(bound, bound)

    def bound_val(self, x):
        """
        given x, return min(threshold, x), in case threshold is not None
        """
        if self.bound < 0:
            return -self.threshold(-x)
        elif self.bound > 0:
            return self.threshold(x)
        return x

    @staticmethod
    def image_feature_shape(img):
        return np.array([img.size(-1), img.size(-2)])

    def project_tensorflow(self, x, y, img_size, img_feat):
        x = torch.clamp(x, min=0, max=img_size[1] - 1)
        y = torch.clamp(y, min=0, max=img_size[0] - 1)

        # it's tedious and contains bugs...
        # when x1 = x2, the area is 0, therefore it won't be processed
        # keep it here to align with tensorflow version
        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()

        max_x_shape = img_feat.shape[1] - 1
        max_y_shape = img_feat.shape[2] - 1
        x1 = torch.clamp(x1, min=0, max=max_x_shape)
        x2 = torch.clamp(x2, min=0, max=max_x_shape)
        y1 = torch.clamp(y1, min=0, max=max_y_shape)
        y2 = torch.clamp(y2, min=0, max=max_y_shape)

        Q11 = img_feat[:, x1, y1].clone()
        Q12 = img_feat[:, x1, y2].clone()
        Q21 = img_feat[:, x2, y1].clone()
        Q22 = img_feat[:, x2, y2].clone()

        weights = torch.mul(x2.float() - x, y2.float() - y)
        Q11 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q11, 0, 1))

        weights = torch.mul(x2.float() - x, y - y1.float())
        Q12 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q12, 0, 1))

        weights = torch.mul(x - x1.float(), y2.float() - y)
        Q21 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q21, 0, 1))

        weights = torch.mul(x - x1.float(), y - y1.float())
        Q22 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q22, 0, 1))

        output = Q11 + Q21 + Q12 + Q22
        return output

    def test(self, resolution, img_features, inputs, poses):
        half_resolution = (resolution - 1) / 2
        camera_c_offset = np.array(self.camera_c) - half_resolution
        # map to [-1, 1]
        # not sure why they render to negative x
        positions = inputs + torch.tensor(
            self.mesh_pos, device=inputs.device, dtype=torch.float
        )
        point_origin = batch_camera_trans_inv(poses[:, 0], positions)

        view_number = poses.size(1)

        h_list = []
        w_list = []

        outputs = []
        for i in range(view_number):
            positions = batch_camera_trans(poses[:, i], point_origin)
            X = positions[:, :, 0]
            Y = positions[:, :, 1]
            Z = positions[:, :, 2]

            w = -self.camera_f[0] * (X / self.bound_val(Z)) + camera_c_offset[0]
            h = self.camera_f[1] * (Y / self.bound_val(Z)) + camera_c_offset[1]
            if self.tensorflow_compatible:
                # to align with tensorflow
                # this is incorrect, I believe
                w += half_resolution[0]
                h += half_resolution[1]

            else:
                # directly do clamping
                w /= half_resolution[0]
                h /= half_resolution[1]

                # clamp to [-1, 1]
                w = torch.clamp(w, min=-1, max=1)
                h = torch.clamp(h, min=-1, max=1)

            feats = []
            for img_feature in img_features[i]:
                feats.append(
                    self.project(resolution, img_feature, torch.stack([w, h], dim=-1))
                )

            output = torch.cat(feats, 2)
            outputs.append(output)

            h_list.append(h)
            w_list.append(w)

        h_view = torch.stack(h_list, dim=1)
        w_view = torch.stack(w_list, dim=1)
        outputs = torch.stack(outputs, dim=1)

        output_max = torch.max(outputs, dim=1)[0]
        output_mean = torch.mean(outputs, dim=1)
        output_std = torch.std(outputs, dim=1)

        return torch.concat([inputs, output_max, output_mean, output_std], dim=2)

        return w_view, h_view, outputs, output_max, output_mean, output_std

    def forward(self, resolution, img_features, inputs, poses):
        half_resolution = (resolution - 1) / 2
        camera_c_offset = np.array(self.camera_c) - half_resolution
        # map to [-1, 1]
        # not sure why they render to negative x
        positions = inputs + torch.tensor(
            self.mesh_pos, device=inputs.device, dtype=torch.float
        )
        point_origin = batch_camera_trans_inv(poses[:, 0], positions)

        view_number = poses.size(1)

        h_list = []
        w_list = []

        outputs = []
        for i in range(view_number):
            positions = batch_camera_trans(poses[:, i], point_origin)
            X = positions[:, :, 0]
            Y = positions[:, :, 1]
            Z = positions[:, :, 2]

            w = -self.camera_f[0] * (X / self.bound_val(Z)) + camera_c_offset[0]
            h = self.camera_f[1] * (Y / self.bound_val(Z)) + camera_c_offset[1]
            if self.tensorflow_compatible:
                # to align with tensorflow
                # this is incorrect, I believe
                w += half_resolution[0]
                h += half_resolution[1]

            else:
                # directly do clamping
                w /= half_resolution[0]
                h /= half_resolution[1]

                # clamp to [-1, 1]
                w = torch.clamp(w, min=-1, max=1)
                h = torch.clamp(h, min=-1, max=1)

            feats = []
            for img_feature in img_features[i]:
                feats.append(
                    self.project(resolution, img_feature, torch.stack([w, h], dim=-1))
                )

            output = torch.cat(feats, 2)
            outputs.append(output)

            h_list.append(h)
            w_list.append(w)

        h_view = torch.stack(h_list, dim=1)
        w_view = torch.stack(w_list, dim=1)
        outputs = torch.stack(outputs, dim=1)

        output_max = torch.max(outputs, dim=1)[0]
        output_mean = torch.mean(outputs, dim=1)
        output_std = torch.std(outputs, dim=1)

        return torch.concat([inputs, output_max, output_mean, output_std], dim=2)

    def project(self, img_shape, img_feat, sample_points):
        """
        :param img_shape: raw image shape
        :param img_feat: [batch_size x channel x h x w]
        :param sample_points: [batch_size x num_points x 2], in range [-1, 1]
        :return: [batch_size x num_points x feat_dim]
        """
        if self.tensorflow_compatible:
            feature_shape = self.image_feature_shape(img_feat)
            points_w = sample_points[:, :, 0] / (img_shape[0] / feature_shape[0])
            points_h = sample_points[:, :, 1] / (img_shape[1] / feature_shape[1])
            output = torch.stack(
                [
                    self.project_tensorflow(
                        points_h[i], points_w[i], feature_shape, img_feat[i]
                    )
                    for i in range(img_feat.size(0))
                ],
                0,
            )
        else:
            output = F.grid_sample(img_feat, sample_points.unsqueeze(1))
            output = torch.transpose(output.squeeze(2), 1, 2)

        return output
