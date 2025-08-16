# general
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, lax

# typing
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from typing import Tuple, Union