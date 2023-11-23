from dataclasses import dataclass

import numpy as np


@dataclass
class HypothesisShape:
    hypothesis_vertices: np.ndarray
    adj_mat: np.ndarray
    num_hypothesis: int
