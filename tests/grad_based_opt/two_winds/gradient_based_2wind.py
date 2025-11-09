import os
from autocvd import autocvd
import optax

multi_gpu = False
if multi_gpu:
    autocvd(num_gpus = 4)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

# numerics
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# jax.config.update("jax_enable_x64", True)

# timing
from timeit import default_timer as timer
from jf1uids.option_classes.simulation_config import FORWARDS, HLL, VARAXIS, XAXIS, YAXIS, ZAXIS
from jf1uids.option_classes.simulation_config import finalize_config

from jax.sharding import PartitionSpec as P, NamedSharding

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from jf1uids import WindParams
from jf1uids._physics_modules._cooling.cooling_options import SimplePowerLawParams, CoolingParams
from jf1uids import SimulationConfig
from jf1uids import get_helper_data
from jf1uids import SimulationParams
from jf1uids import time_integration
from jf1uids.option_classes.simulation_config import BinaryConfig

from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids._physics_modules._stellar_wind.stellar_wind_functions import get_wind_parameters
from jf1uids.option_classes.simulation_config import MUSCL, RK2_SSP, SIMPLE_SOURCE_TERM, SPLIT, UNSPLIT, DONOR_ACCOUNTING
from jf1uids._physics_modules._binary._binary_options import NGP, CIC, TSC
from jf1uids._physics_modules._binary._binary_options import BinaryParams
from jf1uids.option_classes.simulation_config import BinaryConfig

from jf1uids import get_registered_variables
from jf1uids.option_classes import WindConfig
from jf1uids._physics_modules._cooling.cooling_options import CoolingConfig

from jf1uids.option_classes.simulation_config import BACKWARDS, OSHER

# units
from jf1uids import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# wind-specific
from jf1uids._physics_modules._stellar_wind.weaver import Weaver

# turbulence
from jf1uids.initial_condition_generation.turb import create_turb_field
from jf1uids.option_classes.simulation_config import FORWARDS
from jf1uids.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD, OSHER, 
    PERIODIC_BOUNDARY, OPEN_BOUNDARY, REFLECTIVE_BOUNDARY, 
    BoundarySettings, BoundarySettings1D
)

print("👷 Setting up simulation...")

# simulation settings
gamma = 5/3

# spatial domain
box_size = 0.34
num_cells = 64
print("num_cells: ", num_cells)

# activate stellar wind
stellar_wind = True

# turbulence
turbulence = False
wanted_rms = 5 * u.km / u.s

fixed_timestep = False
scale_time = False
dt_max = 0.1
num_timesteps = 1000
boundary=OPEN_BOUNDARY
# setup simulation config
config = SimulationConfig(
    runtime_debugging = False,
    first_order_fallback = True,
    progress_bar = True,
    dimensionality = 3,
    self_gravity_version = SIMPLE_SOURCE_TERM,
    num_ghost_cells = 2,
    box_size = box_size, 
    num_cells = num_cells,
    split = SPLIT,
    limiter = MINMOD,
    binary_config = BinaryConfig(
        binary = False,
        deposit_particles = NGP,  # Options: "ngp", "cic", "tsc"
        central_object_only = False
    ),
    # time_integrator = RK2_SSP,
    time_integrator = MUSCL,
    wind_config = WindConfig(
        stellar_wind = stellar_wind,
        num_injection_cells = 2,
        trace_wind_density = False,
        real_wind_params = False,
    ),
    fixed_timestep = fixed_timestep,
    differentiation_mode = BACKWARDS,
    num_timesteps = num_timesteps,
    return_snapshots = False,
    # num_snapshots = 5,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(
            left_boundary = boundary,
            right_boundary = boundary
        ),
        BoundarySettings1D(
            left_boundary = boundary,
            right_boundary = boundary
        ),
        BoundarySettings1D(
            left_boundary = boundary,
            right_boundary = boundary
        )
    )
)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)


from jf1uids.option_classes.simulation_config import finalize_config


wind_vel_inf = 50.0
mass_loss_rate1 = 1e-11 / 1.309 #2.965e-3 / 1e4 / T_orb * mass_source     #2.965e-3 / 1e6 / T_orb * mass_source
mass_loss_rate2 = 10 * mass_loss_rate1

R_forced = 0.1
mass_source = 1e-4
length_temp = 5 / R_forced
mass_temp = 40 / mass_source
velocity_temp = 2000 / wind_vel_inf
time_temp = length_temp / velocity_temp  #make time_temp as high as possible to have small code time units
code_length = length_temp * u.au 

code_mass = mass_temp * u.M_sun
code_velocity = velocity_temp * u.km / u.s
# code_time = code_length / code_velocity
code_units = CodeUnits(code_length, code_mass, code_velocity)

# time domain
C_CFL = 0.4
t_final = 10 * 1e3 * u.yr
t_end = t_final.to(code_units.code_time).value
dt_max = 0.1 * t_end # not so important if the timestep criterion is good

params = SimulationParams(
    C_cfl = C_CFL,
    dt_max = dt_max,
    gamma = gamma,
    t_end = 0.02, #t_end,

)

# homogeneous initial state
rho_0 = 2 * c.m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

rho_init = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_0.to(code_units.code_density).value
u_init = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
p_init = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * p_0.to(code_units.code_pressure).value

# get initial state
initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho_init,
    velocity_x = u_init,
    velocity_y = u_init,
    velocity_z = u_init,
    gas_pressure = p_init,
)


config = finalize_config(config, initial_state.shape)


import numpy as np
# np.savez_compressed(save_path+'/final_state'+str(num_cells)+'.npz', arr=final_state)

sample_simulation = lambda mass_loss_rate1, mass_loss_rate2: time_integration(initial_state, config, SimulationParams(
    C_cfl=params.C_cfl,
    dt_max=params.dt_max,
    gamma=params.gamma,
    t_end=params.t_end,
    wind_params=WindParams(
        wind_mass_loss_rates = jnp.array([mass_loss_rate1, mass_loss_rate2]),
        wind_final_velocities = jnp.array([wind_vel_inf, wind_vel_inf]),
        wind_injection_positions = jnp.array([[-R_forced/2, 0.0, 0.0],[R_forced/2, 0.0, 0.0]]),
    )
), helper_data, registered_variables)

# generate a reference simulation
reference_params = WindParams(
    wind_mass_loss_rates =  jnp.array([mass_loss_rate1, mass_loss_rate2]),
    wind_final_velocities = jnp.array([wind_vel_inf, wind_vel_inf]), #jnp.array([vel_param, vel_param]) * v_phi0, 
    wind_injection_positions = jnp.array([[-R_forced/2, 0.0, 0.0],[R_forced/2, 0.0, 0.0]])
)

### true parameters
reference_simulation = sample_simulation(
    reference_params.wind_mass_loss_rates[0],
    reference_params.wind_mass_loss_rates[1],
)

np.savez_compressed('ref_final.npz', arr=reference_simulation)
# reference_shock_radius = strongest_shock_radius(reference_simulation, helper_data, 10, 5)


#### i took this from gradient_based_1wind.py  and didnt change the labels everywhere
def slice_density_loss(vel_mass_loss):
    velocity = vel_mass_loss[0]
    mass_loss_rate = vel_mass_loss[1]
    final_state = sample_simulation(velocity, mass_loss_rate)
    level = num_cells // 2
    result = jnp.sum(jnp.abs(final_state[0, :, : , level] - reference_simulation[0, :, : , level]))
    return result

def full_profile_loss(vel_mass_loss):
    velocity = vel_mass_loss[0]
    # jax.debug.print("velocity  {}", velocity)
    mass_loss_rate = vel_mass_loss[1]
    # jax.debug.print("mass_loss_rate {}", mass_loss_rate)
    final_state = sample_simulation(velocity, mass_loss_rate)
    return jnp.sum(jnp.abs(final_state - reference_simulation))

def radial_density_loss(vel_mass_loss):
    level = num_cells // 2
    velocity = vel_mass_loss[0]
    # jax.debug.print("velocity  {}", velocity)
    mass_loss_rate = vel_mass_loss[1]
    # jax.debug.print("mass_loss_rate {}", mass_loss_rate)
    final_state = sample_simulation(velocity, mass_loss_rate)
    final_state_radial_rho = final_state[0, :, :, level]
    reference_radial_rho = reference_simulation[0, :, :, level]
    result = jnp.sum(jnp.abs(final_state_radial_rho - reference_radial_rho))
    jax.debug.print("result {}", result)
    return result


def get_loss_map(mass_loss_rate_range1, mass_loss_rate_range2):
    loss_map = jnp.zeros((len(mass_loss_rate_range1) * len(mass_loss_rate_range2),))
    mass_list1 = jnp.zeros((len(mass_loss_rate_range1) * len(mass_loss_rate_range2),))
    mass_list2 = jnp.zeros((len(mass_loss_rate_range1) * len(mass_loss_rate_range2),))
    ind = 0

    for i, m1 in enumerate(mass_loss_rate_range1):
        for j, m2 in enumerate(mass_loss_rate_range2):
            loss_map = loss_map.at[ind].set(slice_density_loss((m1, m2)))
            mass_list1 = mass_list1.at[ind].set(m1)
            mass_list2 = mass_list2.at[ind].set(m2)
            ind += 1
            print(f"Done {ind}/{len(mass_loss_rate_range1) * len(mass_loss_rate_range2)}")
    return loss_map, mass_list1, mass_list2

# generate a loss map
mass_loss_rates1 = jnp.linspace(
    1e-11 / 1.309 * 0.001,  
    1e-11 / 1.309 * 3.5,    
    60
)

mass_loss_rates2 = jnp.linspace(
    1e-11 / 1.309 * 0.1 * 10,
    1e-11 / 1.309 * 2 * 10,
    60
)

# save_path = os.path.join('grad_opt')

final_state = sample_simulation(mass_loss_rates1[0], mass_loss_rates2[0])
np.savez_compressed('final_state.npz', arr=final_state)
print("Saved final state of sample simulation.")

# create loss map, vel/mass lists
# loss_map, mass_list1, mass_list2 = get_loss_map(mass_loss_rates1, mass_loss_rates2)
# np.savez_compressed('loss_map.npz', arr=loss_map)
# np.savez_compressed('mass_list1.npz', arr=mass_list1)
# np.savez_compressed('mass_list2.npz', arr=mass_list2)

# load loss map, vel/mass lists from previous run
loss_map = np.load('loss_map.npz')['arr']
mass_list1 = np.load('mass_list1.npz')['arr']
mass_list2 = np.load('mass_list2.npz')['arr']


# We pick gradient descent for pedagogical and visualization reasons.
# In practice one would use e.g. Levenberg-Marquardt from the
# optimistix package.
# def gradient_descent_optimization(func, x_init, learning_rate=20, tol=0.5, max_iter=2000):
def any_nonfinite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return any([bool(jnp.any(~jnp.isfinite(leaf))) for leaf in leaves])

def nonfinite_leaves(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return [i for i,l in enumerate(leaves) if bool(jnp.any(~jnp.isfinite(l)))]

def pick_initial_lr(func, x_init, alpha=1e-1, lr_min=1e-1, lr_max=10000.0):
    """
    Heuristic pick for initial learning rate:
      lr = (alpha * ||params||) / (||grad|| + eps)
    with clamping between lr_min and lr_max.
    alpha: desired fraction of param norm to change in first update.
    """
    # compute loss and grads at the start (no AD wrapper needed beyond value_and_grad)
    loss, grads = jax.value_and_grad(func)(x_init)
    # norms (optax.global_norm handles pytrees)
    grad_norm = optax.global_norm(grads)
    param_norm = optax.global_norm(x_init)
    eps = 1e-12
    raw_lr = (alpha * (param_norm + eps)) / (grad_norm + eps)
    # jax.debug.print("pick_initial_lr: raw_lr={} param_norm={} grad_norm={}", raw_lr, param_norm, grad_norm)
    # clamp to reasonable range
    lr = jnp.clip(raw_lr, lr_min, lr_max)
    # turn into Python float for optimizer init etc.
    return float(lr), float(loss), float(param_norm), float(grad_norm)

def pick_initial_lr_numeric(func, x, alpha=1e-1, lr_min=1e-12, lr_max=1.0, eps=1e-20):   #lr_min=1e-12 worked
    # compute numeric grad (central diff)
    grads = numeric_grad(func, x)
    grad_norm = float(optax.global_norm(grads))
    param_norm = float(optax.global_norm(x))
    raw_lr = alpha * (param_norm + eps) / (grad_norm + eps)
    # clamp to sensible range
    print("raw lr:", raw_lr)
    lr = float(max(lr_min, min(lr_max, raw_lr)))
    print(f"pick_initial_lr_numeric -> lr={lr}  param_norm={param_norm:.3e} grad_norm={grad_norm:.3e}")
    return lr


def gradient_descent_optimization(func, x_init, learning_rate=1e-12, tol=5e-13, max_iter=1000, min_iter = 40):
    ### it worked with tol = 1e-12, min_iter = 1
    xlist = []
    x = x_init
    loss_list = []
    xlist.append(x)

    # optimizer = optax.adam(learning_rate=learning_rate)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),    # clip gradients to norm 1.0 (tune)
        optax.adam(learning_rate=learning_rate)  # Adam optimizer
    )
    optimizer_state = optimizer.init(x)

    for it in range(max_iter):
        """
        try:
            loss, f_grad = jax.value_and_grad(lambda z: func(z) + 1e-10 * optax.global_norm(z)**2)(x)
        except Exception as e:
            print("value_and_grad failed at iter", it, "error:", e)
            # fallback to numeric gradient
            f_grad = numeric_grad(func, x)
            loss = float(func(x))
        """
        # detect non-finite grads
        # if any_nonfinite(f_grad):
        #     print("Non-finite gradient at iter", it, "— replacing NaNs with zeros and clipping")
        #     f_grad = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6), f_grad)

        #### directly use numeric grad bc AD seems unstable here
        f_grad = numeric_grad(func, x)
        loss = float(func(x))

        loss_list.append(loss)
        updates, optimizer_state = optimizer.update(f_grad, optimizer_state, params=x)

        # guard updates and x against non-finite
        if any_nonfinite(updates):
            print("Non-finite updates at iter", it, " — zeroing updates")
            updates = jax.tree.map(lambda u: jnp.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6), updates)

        x = optax.apply_updates(x, updates)
        xlist.append(x)
    
        if any_nonfinite(x):
            print("Parameters became non-finite at iter", it, " — aborting")
            break

        # optionally lower learning rate or break on small updates
        update_norm = float(optax.global_norm(updates))
        if not jnp.isfinite(update_norm):
            print("update_norm non-finite -> stopping")
            break

        # jax.debug.print("loss={}, update_norm={}", loss, update_norm)
        if it >= min_iter:
            if update_norm < tol:
                print("Converged at iter", it, "with update_norm", update_norm)
                break
        # if jnp.linalg.norm(updates) < tol:
        #     break

    return x, xlist, loss_list

#### OLD GRADIENT DESCENT CALLS
def gradient_descent_optimization_OLD(func, x_init, learning_rate=1e-12, tol=1e-14, max_iter=500):
    xlist = []
    x = x_init
    loss_list = []
    xlist.append(x)

    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(x)

    for _ in range(max_iter):
        # Compute the function value and its gradient
        loss, f_grad = jax.value_and_grad(func)(x)
        loss_list.append(loss)

        # Update the parameter
        updates, optimizer_state = optimizer.update(f_grad, optimizer_state)
        x = optax.apply_updates(x, updates)
        xlist.append(x)

        # Check convergence
        # if float(optax.global_norm(updates)) < tol:
        if jnp.linalg.norm(updates) < tol:
            break
    return x, xlist, loss_list

### NUMERIC GRAD
def numeric_grad(func, x, eps=1e-12):
    x = jnp.asarray(x)
    g = jnp.zeros_like(x)
    base = float(func(x))
    for i in range(x.size):
        dx = jnp.zeros_like(x).at[i].set(eps)
        f_plus = float(func(x + dx))
        f_minus = float(func(x - dx))
        g = g.at[i].set((f_plus - f_minus) / (2 * eps))
    return g


initial_guess1 = jnp.array([1e-11 / 1.309 * 2, 1e-11 / 1.309 * 15])   ###true: 1e-11 / 1.309, 1e-11 / 1.309 * 10
# initial_guess1 = jnp.array([1e-11 / 1.309 * 1.8, 1e-11 / 1.309 * 10 * 1.3])
# print("suggested lr", lr_suggestion, "loss", loss0, "p_norm", pnorm0, "g_norm", gnorm0)
# x1, xlist1, loss_list1 = gradient_descent_optimization(radial_density_loss, initial_guess1, learning_rate=lr_suggestion)

init1_final_state = sample_simulation(
    initial_guess1[0],
    initial_guess1[1],
)
np.savez_compressed('init1_final_state.npz', arr=init1_final_state)
print("SAVED")

initial_guess2 = jnp.array([1e-11 / 1.309 * 0.3, 1e-11 / 1.309 * 10 * 1.8])
init2_final_state = sample_simulation(
    initial_guess2[0],
    initial_guess2[1],
)
np.savez_compressed('init2_final_state.npz', arr=init2_final_state)
print("SAVED init2")

""" TEST GRADIENT FINITENESS """
"""
loss, f_grad = jax.value_and_grad(slice_density_loss)(initial_guess1)
print('loss finite?', jnp.isfinite(loss), "loss", loss)
print('any grad nonfinite?', any(jnp.any(~jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(f_grad)))

s = sample_simulation(mass_loss_rates1[0], mass_loss_rates2[0])   # forward-only
def stats(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return {i: (float(jnp.nan_to_num(jnp.sum(jnp.isnan(l))).item()), 
                float(jnp.nan_to_num(jnp.sum(jnp.isinf(l))).item()),
                float(jnp.min(l).item()), float(jnp.max(l).item()))
            for i,l in enumerate(leaves)}
print("stats", stats(s))
"""


# learning_rate = pick_initial_lr_numeric(radial_density_loss, initial_guess1)
lr_suggestion, loss0, pnorm0, gnorm0 = pick_initial_lr(slice_density_loss, initial_guess1)
x1, xlist1, loss_list1 = gradient_descent_optimization_OLD(slice_density_loss, initial_guess1)
print("Optimization result:", x1, "xlist:", xlist1, "Loss history:", loss_list1)

initial_guess2 = jnp.array([1e-11 / 1.309 * 0.3, 1e-11 / 1.309 * 10 * 1.8])
# learning_rate = pick_initial_lr_numeric(radial_density_loss, initial_guess2)
lr_suggestion, loss0, pnorm0, gnorm0 = pick_initial_lr(slice_density_loss, initial_guess2)

x2, xlist2, loss_list2 = gradient_descent_optimization_OLD(slice_density_loss, initial_guess2)

initial_guess3 = jnp.array([1e-11 / 1.309 * 2.7, 1e-11 / 1.309 * 10 / 2.5])
lr_suggestion, loss0, pnorm0, gnorm0 = pick_initial_lr(slice_density_loss, initial_guess3)
# learning_rate = pick_initial_lr_numeric(radial_density_loss, initial_guess3)
x3, xlist3, loss_list3 = gradient_descent_optimization_OLD(slice_density_loss, initial_guess3)
print("Done optimization.")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
norm = LogNorm(vmin=loss_map.min(), vmax=loss_map.max())
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.inferno)   #viridis
plt.colorbar(mapper, cax=cax, orientation='vertical', label='Density slice loss')

axs[0].scatter((mass_list1 * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, (mass_list2 * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, c=loss_map, cmap='inferno', norm=norm, s = 15, marker = "s")

axs[0].set_xlabel(r'${\dot M}_1 [M_\odot yr^{-1}]$')
axs[0].set_ylabel(r'${\dot M}_2 [M_\odot yr^{-1}]$')
axs[0].set_title('Loss landscape')

# plot the loss function
axs[1].plot(loss_list1, label='Loss 1', color='blue')

# plot the optimization path
xlist1 = jnp.array(xlist1)
axs[0].plot((xlist1[:, 0] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, (xlist1[:, 1] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, color='blue')
axs[0].scatter(
    [(xlist1[0, 0] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value], [(xlist1[0, 1] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value],
    c='blue', s = 30, label='Initial guess 1'
)


# plot the optimization path
xlist2 = jnp.array(xlist2)
axs[0].plot((xlist2[:, 0] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, (xlist2[:, 1] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, color='purple')
axs[0].scatter(
    [(xlist2[0, 0] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value], [(xlist2[0, 1] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value],
    c='purple', s = 30, label='Initial guess 2'
)

# plot the loss function
axs[1].plot(loss_list2, label='Loss 2', color='purple')

# plot the optimization path
xlist3 = jnp.array(xlist3)
axs[0].plot((xlist3[:, 0] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, (xlist3[:, 1] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, color='red')
axs[0].scatter(
    [(xlist3[0, 0] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value], [(xlist3[0, 1] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value],
    c='red', s = 30, label='Initial guess 3'
)

# plot the loss function
axs[1].plot(loss_list3, label='Loss 3', color='red')


axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Density slice loss')
axs[1].set_title('Loss convergence')
axs[1].set_yscale('log')
axs[1].legend(loc='upper right')
plt.tight_layout()

# mark the true value as a red dot
axs[0].scatter(
    [(mass_loss_rate1 * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value],
    [(mass_loss_rate2 * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value],
    c='white', s = 80, label='True parameters', zorder = 10, edgecolors='black'
)            #marker = "."

axs[0].legend()
axs[1].legend()

fig.savefig('params_opt.png')

#save optimization results
save_path = os.path.join('results')
np.savez_compressed(save_path+'/xlist1.npz', arr=xlist1)
np.savez_compressed(save_path+'/xlist2.npz', arr=xlist2)
np.savez_compressed(save_path+'/xlist3.npz', arr=xlist3)
np.savez_compressed(save_path+'/loss_list1.npz', arr=loss_list1)
np.savez_compressed(save_path+'/loss_list2.npz', arr=loss_list2)
np.savez_compressed(save_path+'/loss_list3.npz', arr=loss_list3)

print("Saved parameter optimization plot.")

