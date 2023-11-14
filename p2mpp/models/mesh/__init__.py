from typing import List
from .ellipsoid import Ellipsoid


def get_base_mesh(mesh: str, mesh_pos: List[float]):
    if mesh == "ellipsoid":
        return Ellipsoid(mesh_pos)
    else:
        raise NotImplementedError(f"No implemented mesh called {mesh} found")
