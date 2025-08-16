from typing import NamedTuple
from jax import numpy as jnp
import jax

class GravitySourceConfig(NamedTuple):
    grav_source: bool = False
    deposition_method: str = None

class GravitySourceParams(NamedTuple):
    source_params: jnp.ndarray = None #jnp.array([0.0, 0.0, 0.0, 1.0])  # [x1, y1, z1, M1]