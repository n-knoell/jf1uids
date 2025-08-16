"""
Fourier-based Poisson solver and simple source term handling
of self gravity. To be improved to an energy-conserving scheme.
"""

# general
from functools import partial
import jax.numpy as jnp
import jax

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Tuple, Union

# fft, in the future use
# https://github.com/DifferentiableUniverseInitiative/JaxPM

# jf1uids data classes
from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import speed_of_sound_crs
from jf1uids._physics_modules._self_gravity._poisson_solver import _compute_gravitational_potential
from jf1uids._riemann_solver._riemann_solver import _riemann_solver
from jf1uids._stencil_operations._stencil_operations import _stencil_add
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import CONSERVATIVE_SOURCE_TERM, LAX_FRIEDRICHS, SIMPLE_SOURCE_TERM, SPLIT, STATE_TYPE_ALTERED, SimulationConfig

# jf1uids constants
from jf1uids.option_classes.simulation_config import FIELD_TYPE, HLL, HLLC, OPEN_BOUNDARY, STATE_TYPE

# jf1uids functions
from jf1uids._geometry.boundaries import _boundary_handler
from jf1uids._riemann_solver.hll import _hll_solver, _hllc_solver
from jf1uids._state_evolution.reconstruction import _reconstruct_at_interface_split, _reconstruct_at_interface_unsplit
from jf1uids.fluid_equations.euler import _euler_flux
from jf1uids.fluid_equations.fluid import conserved_state_from_primitive, primitive_state_from_conserved, speed_of_sound
from jf1uids.option_classes.simulation_params import SimulationParams


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis', 'grid_spacing', 'registered_variables', 'config'])
def _gravitational_source_term_along_axis(
        gravitational_potential: FIELD_TYPE,
        primitive_state: STATE_TYPE,
        grid_spacing: float,
        registered_variables: RegisteredVariables,
        dt: Union[float, Float[Array, ""]],
        gamma: Union[float, Float[Array, ""]],
        config: SimulationConfig,
        params: SimulationParams,
        helper_data: HelperData,
        axis: int,
) -> STATE_TYPE:
    
    """
    Compute the source term for the self-gravity solver along a single axis.
    Currently, simply density * gravitational_acceleration for the momentum 
    and density * velocity * gravitational_acceleration for the energy.

    Args:
        gravitational_potential: The gravitational potential.
        primitive_state: The primitive state.
        grid_spacing: The grid spacing.
        registered_variables: The registered variables.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        helper_data: The helper data.
        axis: The axis along which to compute the source term.

    Returns:
        The source term.
    
    """

    rho = primitive_state[registered_variables.density_index]
    v_axis = primitive_state[axis]

    # a_i = - (phi_{i+1} - phi_{i-1}) / (2 * dx)
    acceleration = -_stencil_add(gravitational_potential, indices = (1, -1), factors = (1.0, -1.0), axis = axis - 1) / (2 * grid_spacing)
    # it is axis - 1 because the axis is 1-indexed as usually the zeroth axis are the different
    # fields in the state vector not the spatial dimensions, but here we only have the spatial dimensions

    source_term = jnp.zeros_like(primitive_state)

    # set momentum source
    source_term = source_term.at[axis].set(rho * acceleration)

    if config.self_gravity_version == SIMPLE_SOURCE_TERM:

        # set energy source
        source_term = source_term.at[registered_variables.pressure_index].set(rho * v_axis * acceleration)

    elif config.self_gravity_version == CONSERVATIVE_SOURCE_TERM:
        # ===============================================

        # better energy source
        if config.first_order_fallback:
            primitive_state_left = jnp.roll(primitive_state, shift = 1, axis = axis)
            primitive_state_right = primitive_state
        else:
            if config.split == SPLIT:
                primitive_state_left, primitive_state_right = _reconstruct_at_interface_split(primitive_state, dt, gamma, config, helper_data, registered_variables, axis)
            else:
                # TODO: improve efficiency
                # this is currently suboptimal, the reconstruction is done for all axes
                # but we only need it for the current axis
                primitives_left_interface, primitives_right_interface = _reconstruct_at_interface_unsplit(
                    primitive_state,
                    dt,
                    gamma,
                    config,
                    params,
                    helper_data,
                    registered_variables
                )
                primitive_state_left = primitives_left_interface[axis - 1]
                primitive_state_right = primitives_right_interface[axis - 1]
        
        if config.riemann_solver == LAX_FRIEDRICHS:

            # TODO: analogous split for the other solvers

            conserved_left = conserved_state_from_primitive(primitive_state_left, gamma, config, registered_variables)
            conserved_right = conserved_state_from_primitive(primitive_state_right, gamma, config, registered_variables)

            # alpha = jnp.max(jnp.maximum(jnp.abs(u_L) + c_L, jnp.abs(u_R) + c_R))
            u = primitive_state[axis]
            rho = primitive_state[registered_variables.density_index]
            p = primitive_state[registered_variables.pressure_index]
            c = speed_of_sound(rho, p, gamma)
            alpha = jnp.max(jnp.abs(u) + c)

            fluxes_left = _euler_flux(primitive_state_left, gamma, config, registered_variables, axis)
            fluxes_right = _euler_flux(primitive_state_right, gamma, config, registered_variables, axis)

            # fluxes = 0.5 * (fluxes_left + fluxes_right) - 0.5 * alpha * (conserved_right - conserved_left)

            # what cell i accounts for regarding the flux between i-1 and i
            fluxes_i_to_im1 = 0.5 * fluxes_right + jnp.minimum(- 0.5 * alpha * (conserved_right - conserved_left), 0)

            # what cell i-1 accounts for regarding the flux between i-1 and i
            fluxes_im1 = 0.5 * fluxes_left + jnp.maximum(- 0.5 * alpha * (conserved_right - conserved_left), 0)
            fluxes_i_to_ip1 = jnp.roll(fluxes_im1, shift = -1, axis = axis)
        else:
        
            # at index i, the fluxes array contains the flux from i-1 to i
            fluxes = _riemann_solver(primitive_state_left, primitive_state_right, primitive_state, gamma, config, registered_variables, axis)
            fluxes_i_to_ip1 = jnp.maximum(jnp.roll(fluxes, shift = -1, axis = axis), 0)
            fluxes_i_to_im1 = jnp.minimum(fluxes, 0)

        acc_backward = -_stencil_add(gravitational_potential, indices = (0, -1), factors = (1.0, -1.0), axis = axis - 1) / grid_spacing
        acc_forward = -_stencil_add(gravitational_potential, indices = (1, 0), factors = (1.0, -1.0), axis = axis - 1) / grid_spacing

        fluxes_acc = fluxes_i_to_im1 * acc_backward + fluxes_i_to_ip1 * acc_forward

        source_term = source_term.at[registered_variables.pressure_index].set(fluxes_acc[0])

        # ===============================================

    return source_term

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['grid_shape', 'grid_spacing'])
def _deposit_particles_ngp(
    particle_positions: Float[Array, "n 3"],
    particle_masses:    Union[Float[Array, ""], Float[Array, "n"]],
    grid_shape:         Tuple[int, int, int],
    grid_spacing:       float
) -> Float[Array, "nx ny nz"]:
    """
    Deposit n point-masses to nearest grid cell (NGP).
    Positions in same units as grid, origin at (0,0,0).
    """
    grid_extent = jnp.array(grid_shape) * grid_spacing    # [Nx*dx, Ny*dy, Nz*dz]
    grid_min    = -0.5 * grid_extent                     
    idx = jnp.floor_divide(particle_positions - grid_min, grid_spacing).astype(int)
    idx = jnp.clip(idx, 0, jnp.array(grid_shape) - 1)    
    flat_idx = idx[:,0] * (grid_shape[1]*grid_shape[2]) + idx[:,1] * grid_shape[2] + idx[:,2]
    rho_flat = jnp.zeros(grid_shape[0]*grid_shape[1]*grid_shape[2])
    rho_flat = rho_flat.at[flat_idx].add(particle_masses)
    return rho_flat.reshape(grid_shape)

### Cloud-In-Cell (CIC) particle deposition (might be better????)
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=('grid_shape', 'grid_spacing'))
def _deposit_particles_cic(
    particle_positions: Float[Array, "n 3"],
    particle_masses:    Union[Float[Array, ""], Float[Array, "n"]],
    grid_shape:         Tuple[int, int, int],
    grid_spacing:       float
) -> Float[jnp.ndarray, "nx ny nz"]:
    """
    Cloud-In-Cell (CIC) deposit (3D).

    particle_positions: (N,3) world coords
    particle_masses:    (N,)
    grid_shape:         (nx,ny,nz) -- MUST be a Python tuple at call time (static)
    grid_spacing:       scalar dx (treated as static here)
    """
    particle_masses = jnp.atleast_1d(particle_masses)
    nx, ny, nz = grid_shape
    grid_extent = jnp.array([nx, ny, nz]) * grid_spacing
    grid_min = -0.5 * grid_extent
    # relative continuous index in grid coordinates
    rel = (particle_positions - grid_min) / grid_spacing   # (N,3)
    i0 = jnp.floor(rel).astype(jnp.int32)                 # lower index (N,3)
    f  = rel - i0.astype(rel.dtype)                       # fractional part (N,3) in [0,1)
    # 8 neighbor offsets for CIC (cartesian product of {0,1}^3)
    offsets = jnp.array([
        [0,0,0],[0,0,1],[0,1,0],[0,1,1],
        [1,0,0],[1,0,1],[1,1,0],[1,1,1],
    ], dtype=jnp.int32)                                   # (8,3)

    # neighbor indices (N,8,3)
    neigh_idx = i0[:, None, :] + offsets[None, :, :]
    # clip indices to grid boundaries (non-periodic)
    max_idx = jnp.array([nx - 1, ny - 1, nz - 1], dtype=jnp.int32)
    neigh_idx = jnp.clip(neigh_idx, 0, max_idx)
    # weights: for each dim weight is (1-f) if offset==0 else f; multiply over dims -> (N,8)
    f_b = f[:, None, :]                                   # (N,1,3)
    # boolean mask of offsets==0 broadcasted -> choose (1-f) or f
    w_comp = jnp.where(offsets[None, :, :] == 0, 1.0 - f_b, f_b)  # (N,8,3)
    weights = jnp.prod(w_comp, axis=-1)                   # (N,8)
    # linearize 3D indices to flat indices (row-major: x*(ny*nz) + y*nz + z)
    flat_idx = (neigh_idx[..., 0] * (ny * nz)
                + neigh_idx[..., 1] * nz
                + neigh_idx[..., 2])                      # (N,8)
    # flatten for scatter
    flat_idx_flat = flat_idx.reshape(-1)                  # (N*8,)
    values_flat = (particle_masses[:, None] * weights).reshape(-1)  # (N*8,)
    n_cells = nx * ny * nz                                # Python int (static)
    rho_flat = jnp.zeros(n_cells, dtype=particle_masses.dtype)
    rho_flat = rho_flat.at[flat_idx_flat].add(values_flat)

    return rho_flat.reshape((nx, ny, nz))

# Triangular-Shaped-Cloud (TSC) particle deposition (quadratic B-spline)
# This is a more accurate method than CIC, but more expensive. It spreads each particle’s mass to the
#  3 nearest grid points along each axis (3×3×3 = 27 cells in 3D) with quadratic weights.
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=('grid_shape', 'grid_spacing'))
def _deposit_particles_tsc(
    particle_positions: Float[Array, "n 3"],
    particle_masses:    Union[Float[Array, ""], Float[Array, "n"]],
    grid_shape:         Tuple[int, int, int],
    grid_spacing:       float
) -> Float[jnp.ndarray, "nx ny nz"]:
    """
    TSC (Triangular-Shaped-Cloud) deposit in 3D.

    - particle_positions: (N,3) world coordinates
    - particle_masses:    (N,)
    - grid_shape:         (nx,ny,nz) as a Python tuple (must be static at call time)
    - grid_spacing:       scalar dx (treated as static here)
    Returns: rho (nx,ny,nz) with total mass conserved.
    """
    particle_masses = jnp.atleast_1d(particle_masses)
    nx, ny, nz = grid_shape
    grid_extent = jnp.array([nx, ny, nz]) * grid_spacing
    grid_min = -0.5 * grid_extent   # center the grid
    # continuous position in grid units (index-space)
    rel = (particle_positions - grid_min) / grid_spacing    # (N,3)
    # floor(rel) gives a central index; neighbors are floor(rel)-1, floor(rel), floor(rel)+1
    i_center = jnp.floor(rel).astype(jnp.int32)             # (N,3)
    # Offsets for TSC: cartesian product of [-1,0,1]^3 -> 27 neighbors
    offsets = jnp.array([[i, j, k]
                         for i in (-1, 0, 1)
                         for j in (-1, 0, 1)
                         for k in (-1, 0, 1)], dtype=jnp.int32)   # (27,3)
    # neighbor indices (N,27,3)
    neigh_idx = i_center[:, None, :] + offsets[None, :, :]   # (N,27,3)
    # Clip indices to grid bounds (non-periodic behaviour)
    max_idx = jnp.array([nx - 1, ny - 1, nz - 1], dtype=jnp.int32)
    neigh_idx = jnp.clip(neigh_idx, 0, max_idx)
    # compute 1D distances 
    r = rel[:, None, :] - neigh_idx.astype(rel.dtype)       # (N,27,3)
    s = jnp.abs(r)                                          # (N,27,3)
    # 1D TSC kernel evaluated vectorized:
    def W1D_from_s(s_component):
        w = jnp.where(s_component <= 0.5,
                      0.75 - s_component**2,
                      jnp.where(s_component <= 1.5,
                                0.5 * (1.5 - s_component)**2,
                                0.0))
        return w

    wx = W1D_from_s(s[..., 0])   # (N,27)
    wy = W1D_from_s(s[..., 1])   # (N,27)
    wz = W1D_from_s(s[..., 2])   # (N,27)
    weights = wx * wy * wz      # (N,27)
    # Flatten neighbor flat indices and weighted mass values for scatter
    flat_idx = (neigh_idx[..., 0] * (ny * nz)
                + neigh_idx[..., 1] * nz
                + neigh_idx[..., 2])                    # (N,27)

    flat_idx_flat = flat_idx.reshape(-1)                 # (N*27,)
    values_flat = (particle_masses[:, None] * weights).reshape(-1)  # (N*27,)
    n_cells = nx * ny * nz    # Python int (static)
    rho_flat = jnp.zeros(n_cells, dtype=particle_masses.dtype)
    rho_flat = rho_flat.at[flat_idx_flat].add(values_flat)

    return rho_flat.reshape((nx, ny, nz))



@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _apply_self_gravity(
    primitive_state: STATE_TYPE,
    old_primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    gamma: Union[float, Float[Array, ""]],
    gravitational_constant: Union[float, Float[Array, ""]],
    dt: Union[float, Float[Array, ""]]
) -> STATE_TYPE:

    rho_gas = old_primitive_state[registered_variables.density_index]

    if config.gravity_source_config.grav_source:
        source_params = params.gravity_source_params.source_params
        particle_positions = source_params[:3].reshape(-1, 3)
        particle_masses = source_params[3]  
        grid_shape   = rho_gas.shape
        grid_spacing = config.grid_spacing
        if config.gravity_source_config.deposition_method == "ngp":
            rho_source = _deposit_particles_ngp(particle_positions, particle_masses, grid_shape, grid_spacing)
        elif config.gravity_source_config.deposition_method == "tsc":
            rho_source = _deposit_particles_tsc(particle_positions, particle_masses, grid_shape, grid_spacing)
        else:
            raise ValueError(f"Unknown deposit_particles method: {config.gravity_source_config.deposition_method}")


        """
        # --- debug: print nonzero deposits (idx, flat idx, world pos, and mass) ---
        # caution: can be large if many nonzero cells -> limit with max_print
        max_print = 50

        # indices of non-zero cells (returns tuple of arrays (i_arr, j_arr, k_arr)) 
        nz = jnp.where(rho_source > 0)

        # stack to shape (n_nonzero, 3)
        idxs = jnp.stack(nz, axis=-1)        # shape (count, 3) or shape (0,3) if none

        # flat index for each non-zero cell
        nx, ny, nzdim = grid_shape
        flat_idx = idxs[:, 0] * (ny * nzdim) + idxs[:, 1] * nzdim + idxs[:, 2]

        # grid -> world coordinates (cell centers)
        grid_extent = jnp.array(grid_shape) * grid_spacing
        grid_min = -0.5 * grid_extent
# idxs is shape (count,3); adding 0.5 gives center of cell
        world_pos = grid_min + (idxs + 0.5) * grid_spacing  # shape (count,3)

# masses in those cells
        masses = rho_source[idxs[:, 0], idxs[:, 1], idxs[:, 2]]  # shape (count,)

# how many non-zero deposits
        count = idxs.shape[0]

# limit how many entries we actually print
        n_to_print = jnp.minimum(count, max_print)

# take first n_to_print entries
        first_idxs = idxs[:n_to_print]
        first_flat = flat_idx[:n_to_print]
        first_pos = world_pos[:n_to_print]
        first_masses = masses[:n_to_print]

# print a short summary and first entries (note: placeholders must match args)
        jax.debug.print("Found {} non-zero deposit cells; printing first {} entries", count, n_to_print)
        jax.debug.print("grid_idx (i,j,k): {}", first_idxs)
        jax.debug.print("flat_idx: {}", first_flat)
        jax.debug.print("world_pos (x,y,z) of cell centers: {}", first_pos)
        jax.debug.print("masses in those cells: {}", first_masses)
# --- end debug ---
"""



        rho_tot = rho_gas + rho_source
    else:
        rho_tot = rho_gas 

    potential = _compute_gravitational_potential(rho_tot, config.grid_spacing, config, gravitational_constant)

    source_term = jnp.zeros_like(primitive_state)

    for i in range(config.dimensionality):
        source_term = source_term + _gravitational_source_term_along_axis(
                                        potential,
                                        old_primitive_state,
                                        config.grid_spacing,
                                        registered_variables,
                                        dt,
                                        gamma,
                                        config,
                                        params,
                                        helper_data,
                                        i + 1
                                    )

    conserved_state = conserved_state_from_primitive(primitive_state, gamma, config, registered_variables)

    conserved_state = conserved_state + dt * source_term

    primitive_state = primitive_state_from_conserved(conserved_state, gamma, config, registered_variables)

    primitive_state = _boundary_handler(primitive_state, config)

    return primitive_state