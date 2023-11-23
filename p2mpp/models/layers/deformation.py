import torch
import torch.nn as nn


class SampleHypothesis(nn.Module):
    def __init__(self, sample_delta: torch.Tensor):
        super(SampleHypothesis, self).__init__()
        self.sample_delta = sample_delta

    def forward(self, mesh_coords: torch.Tensor):
        """
        Local Grid Sample for fast matching init mesh
        :param mesh_coords:
        [N,S,3] ->[NS,3] for projection
        :return: sample_points_per_vertices: [NS, 3]
        """

        batch_size = mesh_coords.size(0)
        center_points = mesh_coords.unsqueeze(2)
        center_points = center_points.repeat(1, 1, 43, 1)

        delta = self.sample_delta.unsqueeze(0)

        outputs = center_points + delta

        outputs = outputs.view(batch_size, -1, 3)
        return outputs
