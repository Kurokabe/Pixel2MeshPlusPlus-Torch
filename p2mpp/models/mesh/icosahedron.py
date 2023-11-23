import pickle

import config
from p2mpp.models.mesh.hypothesis_shape import HypothesisShape


class Icosahedron(HypothesisShape):
    def __init__(self, file=config.ICOSAHEDRON_PATH):
        with open(file, "rb") as f:
            data = pickle.load(f, encoding="bytes")
        self.hypothesis_vertices = data["sample_coord"]
        self.adj_mat = data["sample_cheb_dense"][1]

        self.num_hypothesis = self.hypothesis_vertices.shape[0]
