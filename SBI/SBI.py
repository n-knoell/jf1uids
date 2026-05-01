import os
from autocvd import autocvd
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
autocvd(num_gpus = 1, interval = 1)

# numerics
import jax
import jax.numpy as jnp
from jax import random, jit
import numpy as np
from scipy.interpolate import interp1d

#Halpha
from Halpha import project_theta_pi2_fast
from Halpha import build_j_map
from Halpha import sb_to_photons_code, add_detector_noise_jax

import torch
import pickle
from sbi.inference import NPE
from sbi.utils import BoxUniform
import torch.nn as nn
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
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids._physics_modules._stellar_wind.stellar_wind_functions import get_wind_parameters
from jf1uids.option_classes.simulation_config import MUSCL, RK2_SSP, SIMPLE_SOURCE_TERM, SPLIT, UNSPLIT, DONOR_ACCOUNTING

from jf1uids import get_registered_variables
from jf1uids.option_classes import WindConfig
from jf1uids._physics_modules._cooling.cooling_options import CoolingConfig

from jf1uids.option_classes.simulation_config import BACKWARDS, OSHER
from jf1uids._physics_modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig, TurbulentForcingParams


from jf1uids._physics_modules._binary._binary_options import NGP, CIC, TSC
from jf1uids._physics_modules._binary._binary_options import BinaryParams
from jf1uids.option_classes.simulation_config import BinaryConfig
from jf1uids._physics_modules._binary._binary import  binary_starting_orbits_at_phase
# units
from jf1uids import CodeUnits
from astropy import units as u
import astropy.constants as c

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

# jax.config.update("jax_enable_x64", True)

print("👷 Setting up simulation...")

# simulation settings
gamma = 5/3

# spatial domain
box_size = 0.34
num_cells = 64
print("num_cells: ", num_cells)

# activate stellar wind
stellar_wind = True
binary_option = True
turbulent_forcing = True
num_snaps = 11

fixed_timestep = False
scale_time = False
dt_max = 0.1
num_timesteps = 1600
boundary=OPEN_BOUNDARY
# setup simulation config
config = SimulationConfig(
    runtime_debugging = False,
    first_order_fallback = True,
    progress_bar = True,
    dimensionality = 3,
    num_ghost_cells = 2,
    box_size = box_size, 
    num_cells = num_cells,
    split = SPLIT,
    limiter = MINMOD,
    # time_integrator = RK2_SSP,
    time_integrator = MUSCL,
    cooling_config = CoolingConfig(
        cooling = False,
    ),
    wind_config = WindConfig(
        stellar_wind = stellar_wind,
        num_injection_cells = 2,
        trace_wind_density = False,
        real_wind_params = False,
    ),
    binary_config = BinaryConfig(
        binary = binary_option,
        deposit_particles = NGP,  # Options: "ngp", "cic", "tsc"
        central_object_only = False
    ),
    turbulent_forcing_config = TurbulentForcingConfig(
        turbulent_forcing = turbulent_forcing,
    ),
    fixed_timestep = fixed_timestep,
    differentiation_mode = FORWARDS,
    num_timesteps = num_timesteps,
    return_snapshots = True,
    num_snapshots = num_snaps,
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

C_CFL = 0.4
R_forced = 0.1
sep_in_au = 10  #upper-limit: 80, lower-limit: 5
length_temp = sep_in_au / 0.1  # before
mass_temp = 0.01 / 1  #1e-4
code_length = length_temp * u.au 
code_mass = mass_temp * u.M_sun
code_velocity = np.sqrt(c.G * code_mass / code_length).to(u.km/u.s)     # code_velocity = velocity_temp * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)
a = box_size / 3.8 # semi-major axis

# homogeneous initial state
rho_0 = 2 * c.m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

print("p0",p_0.to(code_units.code_pressure).value)
print("p0",(p_0.to(code_units.code_pressure)).to(u.Pa))
rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_0.to(code_units.code_density).value

print("rho0",rho_0.to(code_units.code_density).value)
print("rho0",(rho_0.to(code_units.code_density)).to(u.g / u.cm**3))
m_p = c.m_p.to(code_units.code_mass).value
print("m_p",m_p)
n_e = 0.7 * (rho_0.to(code_units.code_density).value) / m_p  / code_units.code_length**3
print("n_e",n_e)

u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))

x = jnp.linspace(0, config.box_size, config.num_cells)
y = jnp.linspace(0, config.box_size, config.num_cells)
z = jnp.linspace(0, config.box_size, config.num_cells)

p = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * p_0.to(code_units.code_pressure).value

# construct primitive state
initial_state = construct_primitive_state(
    config = config,
    registered_variables=registered_variables,
    density = rho,
    velocity_x = u_x,
    velocity_y = u_y,
    velocity_z = u_z,
    gas_pressure = p
)

config = finalize_config(config, initial_state.shape)

mass_array = np.array([12,15,20,25,32,40,60,85,120])
mlr0_array = np.load("mlr0_array.npy")
print("mlr0_array: ", mlr0_array)

mass_array_code = (mass_array * u.M_sun).to(code_units.code_mass).value
mlr0_code = (mlr0_array * u.M_sun / u.yr).to(code_units.code_mass / code_units.code_time).value

mlr_values = jnp.array(mlr0_code)
mass_values = jnp.array(mass_array_code)

cell_size = config.box_size / config.num_cells

T5yr = (5 * u.yr).to(code_units.code_time).value
T3yr = (3 * u.yr).to(code_units.code_time).value
print("T5yr in code time: ", T5yr)

def _just_hydro_result(theta, key): 
    mlr1 = jnp.exp(theta[0])   # mass loss rate 1
    mlr2 = jnp.exp(theta[1])  # mass loss rate 2
    v_inf1 = theta[2]     # wind terminal velocity 1
    v_inf2 = theta[3]   

    e = theta[4]
    inc = jnp.arccos(theta[5]) * (180.0 / jnp.pi)  # convert back to deg
    turbulence_strength = theta[6]

    mass_source1_mean = jnp.interp(mlr1, mlr_values, mass_values)
    mass_source2_mean = jnp.interp(mlr2, mlr_values, mass_values)
    mass_source1 = mass_source1_mean + 0.05 * mass_source1_mean * random.normal(key)
    key = random.split(key)[0]
    mass_source2 = mass_source2_mean + 0.05 * mass_source2_mean * random.normal(key)

    phi0 = 0.0
    masses = jnp.array([mass_source1, mass_source2])
    orbits = binary_starting_orbits_at_phase(a, e, inc, mass_source1, mass_source2, phi0, true_anom_deg=0.0)

    wind_luminosity = 0.5 * (mlr1 + mlr2) / 2 * ((v_inf1 + v_inf2) / 2)**2
    energy_injection_rate = turbulence_strength * wind_luminosity

    result = time_integration(initial_state, config, SimulationParams(
    C_cfl=C_CFL,
    dt_max=dt_max,
    gamma=gamma,
    t_end=T5yr, 
    wind_params=WindParams(
        wind_mass_loss_rates = jnp.array([mlr1, mlr2]),
        wind_final_velocities = jnp.array([v_inf1, v_inf2]),
    ),
    turbulent_forcing_params = TurbulentForcingParams(
        energy_injection_rate = energy_injection_rate
    ),
    binary_params = BinaryParams(
        masses = masses,
        binary_state = orbits
    )
    ), helper_data, registered_variables, key = key)
    
    return result

# @jit
def sample_simulation(theta, key): 
    mlr1 = jnp.exp(theta[0])   # mass loss rate 1
    mlr2 = jnp.exp(theta[1])  # mass loss rate 2
    v_inf1 = theta[2]     # wind terminal velocity 1
    v_inf2 = theta[3]   

    e = theta[4]
    inc = jnp.arccos(theta[5]) * (180.0 / jnp.pi)  # convert back to deg
    turbulence_strength = theta[6]

    mass_source1_mean = jnp.interp(mlr1, mlr_values, mass_values)
    mass_source2_mean = jnp.interp(mlr2, mlr_values, mass_values)
    mass_source1 = mass_source1_mean + 0.05 * mass_source1_mean * random.normal(key)
    key = random.split(key)[0]
    mass_source2 = mass_source2_mean + 0.05 * mass_source2_mean * random.normal(key)

    phi0 = 0.0
    masses = jnp.array([mass_source1, mass_source2])
    orbits = binary_starting_orbits_at_phase(a, e, inc, mass_source1, mass_source2, phi0, true_anom_deg=0.0)

    wind_luminosity = 0.5 * (mlr1 + mlr2) / 2 * ((v_inf1 + v_inf2) / 2)**2
    energy_injection_rate = turbulence_strength * wind_luminosity

    result = time_integration(initial_state, config, SimulationParams(
    C_cfl=C_CFL,
    dt_max=dt_max,
    gamma=gamma,
    t_end=T5yr,
    wind_params=WindParams(
        wind_mass_loss_rates = jnp.array([mlr1, mlr2]),
        wind_final_velocities = jnp.array([v_inf1, v_inf2]),
    ),
    turbulent_forcing_params = TurbulentForcingParams(
        energy_injection_rate = energy_injection_rate
    ),
    binary_params = BinaryParams(
        masses = masses,
        binary_state = orbits
    )
    ), helper_data, registered_variables, key = key)

    snap_keys = random.split(key, num_snaps)
    def _process_one(state, rng_key):
        density = state[0]
        pressure = state[4]
        j_map = build_j_map(density, pressure)
        J_map = project_theta_pi2_fast(j_map, cell_size)
        sky_photons = sb_to_photons_code(J_map)
        sky_photons = jnp.clip(sky_photons, a_min=1.0, a_max=None) 
        # clean photon-count images and draw a fresh noise realisation at training
        # time (see `add_fresh_noise` in retrain_temporal.py).
        return sky_photons

    converted_images = jax.vmap(_process_one)(result.states, snap_keys)  # shape: (num_snapshots, H, W)
    return converted_images

mlr_lower = 8e-9 * u.M_sun / u.yr            #
mlr_upper = 1e-5 * u.M_sun / u.yr
wind_vel_lower = 1200 * u.km / u.s 
wind_vel_upper = 3200 * u.km / u.s   

mlr_lower = mlr_lower.to(code_units.code_mass / code_units.code_time).value
mlr_upper = mlr_upper.to(code_units.code_mass / code_units.code_time).value
wind_vel_lower = wind_vel_lower.to(code_units.code_velocity).value
wind_vel_upper = wind_vel_upper.to(code_units.code_velocity).value

print("mlr_lower code: ", mlr_lower, " mlr_upper code: ", mlr_upper, " wind_vel_lower code: ", wind_vel_lower, " wind_vel_upper code: ", wind_vel_upper)
log_mlr_lower = jnp.log(mlr_lower)
log_mlr_upper = jnp.log(mlr_upper)
print("log_mlr_lower code: ", log_mlr_lower, " log_mlr_upper code: ", log_mlr_upper)
log_mlr_lower = float(log_mlr_lower)
log_mlr_upper = float(log_mlr_upper)
wind_vel_lower = float(wind_vel_lower)
wind_vel_upper = float(wind_vel_upper)

e_lower = 0.0
e_upper = 0.85
cos_inc_lower = 0.0
cos_inc_upper = 1.0

turbulence_strength_lower = 0.0
turbulence_strength_upper = 0.025

prior = BoxUniform(
    low=torch.tensor([log_mlr_lower, log_mlr_lower, wind_vel_lower, wind_vel_lower, e_lower, cos_inc_lower, turbulence_strength_lower]),
    high=torch.tensor([log_mlr_upper, log_mlr_upper, wind_vel_upper, wind_vel_upper, e_upper, cos_inc_upper, turbulence_strength_upper])
)    
# print("Prior", prior)
print("Prior high: ", prior.high)    
print("Prior low: ", prior.low)    

from matplotlib.colors import LogNorm

def main():
    # plt.rcParams.update({
    #     'axes.labelsize': 14,   # x and y labels
    #     'axes.titlesize': 16,   # plot title
    #     'xtick.labelsize': 12,  # x tick labels
    #     'ytick.labelsize': 12,  # y tick labels
    #     'legend.fontsize': 12,  # legend text
    # })

    # simulate data
    batched_simulator_jax = jax.vmap(sample_simulation)
    Num_sims = 4500
    chunk = 150  # number of simulations to run in each batch
    theta = prior.sample((Num_sims,)) 
    theta_jax = jnp.asarray(theta.numpy())

    fails = 0
    total = Num_sims / chunk

    def run_in_chunks(thetas, chunk):
        outs = []
        used_thetas = []
        for i in range(0, thetas.shape[0], chunk):
            random_int = np.random.randint(0, 1e6)
            universal_key = jax.random.PRNGKey(random_int)
            split_keys = jax.random.split(universal_key, num=chunk)
            chunk_theta = thetas[i:i+chunk]
            outs.append(batched_simulator_jax(chunk_theta, split_keys))
            used_thetas.append(chunk_theta)
        return jnp.concatenate(outs, axis=0), jnp.concatenate(used_thetas, axis=0)

    start = timer()
    x_jax, used_thetas = run_in_chunks(theta_jax, chunk=chunk)
    x = torch.from_numpy(np.asarray(x_jax).copy())
    theta_conv = torch.from_numpy(np.asarray(used_thetas).copy())

    # save the simulations (clean photon counts; noise is added at training time)
    torch.save({"theta": theta_conv, "x": x}, "simulations_clean.pt")
    print("Saved simulations (clean photon counts)")
    print(f"Number of failed simulations: {fails} out of {total}")
    end = timer()
    print(f"Time taken for {Num_sims} simulations: {end - start:.2f} seconds")


    ### REFERENCE SIMULATIONS AND PLOTS (visualization) ###
    # temp =  6.5e-8 * u.M_sun / u.yr       
    # temp2 = 1.5e-8 * u.M_sun / u.yr 
    # mlr1 = temp.to(code_units.code_mass / code_units.code_time).value
    # mlr2 = temp2.to(code_units.code_mass / code_units.code_time).value
    # log_mlr = float(jnp.log(mlr1))
    # log_mlr2 = float(jnp.log(mlr2))
    # wind_vel1 = 2010 * u.km / u.s
    # wind_vel2 = 1680 * u.km / u.s
    # wind_vel1 = wind_vel1.to(code_units.code_velocity).value
    # wind_vel2 = wind_vel2.to(code_units.code_velocity).value

    # theta_low = torch.tensor([log_mlr, log_mlr2, wind_vel1, wind_vel2, 0.21, 0.35, 0.011])
    # print("Reference theta (log_mlr1, log_mlr2, v_inf1, v_inf2, e, cos_inc, turbulence_strength): ", theta_low)
    
    # make_reference_observation(theta_low, "low")

    # temp =  8.4e-6 * u.M_sun / u.yr       
    # temp2 = 1.2e-8 * u.M_sun / u.yr 
    # mlr1 = temp.to(code_units.code_mass / code_units.code_time).value
    # mlr2 = temp2.to(code_units.code_mass / code_units.code_time).value
    # log_mlr = float(jnp.log(mlr1))
    # log_mlr2 = float(jnp.log(mlr2))
    # wind_vel1 = 3080 * u.km / u.s
    # wind_vel2 = 1740 * u.km / u.s
    # wind_vel1 = wind_vel1.to(code_units.code_velocity).value
    # wind_vel2 = wind_vel2.to(code_units.code_velocity).value

    # theta_diff = torch.tensor([log_mlr, log_mlr2, wind_vel1, wind_vel2, 0.09, 0.89, 0.007])
    
    # make_reference_observation(theta_diff, "diff")

    # temp =  4e-6 * u.M_sun / u.yr       
    # temp2 = 4e-6 * u.M_sun / u.yr 
    # mlr1 = temp.to(code_units.code_mass / code_units.code_time).value
    # mlr2 = temp2.to(code_units.code_mass / code_units.code_time).value
    # log_mlr = float(jnp.log(mlr1))
    # log_mlr2 = float(jnp.log(mlr2))
    # wind_vel1 = 2977 * u.km / u.s
    # wind_vel2 = 2505 * u.km / u.s
    # wind_vel1 = wind_vel1.to(code_units.code_velocity).value
    # wind_vel2 = wind_vel2.to(code_units.code_velocity).value

    # theta_high = torch.tensor([log_mlr, log_mlr2, wind_vel1, wind_vel2, 0.82, 0.62, 0.016])
    # make_reference_observation(theta_high, "high")

    # name = "high"
    # data = np.load(f"ref_obs_{name}.npz", allow_pickle=True)
    # one_run = data["one_run"]
    # theta_jax = data["theta"]
    # _do_plots(one_run,name)

    # name = "low"
    # data = np.load(f"ref_obs_{name}.npz", allow_pickle=True)
    # one_run = data["one_run"]
    # theta_jax = data["theta"]
    # _do_plots(one_run,name)

    # name = "diff"
    # data = np.load(f"ref_obs_{name}.npz", allow_pickle=True)
    # one_run = data["one_run"]
    # theta_jax = data["theta"]
    # _do_plots(one_run,name)

    pass


def make_reference_observation(theta, name):
    theta_jax = jnp.asarray(theta.numpy())
    one_run = sample_simulation(theta_jax, random.PRNGKey(0))
    np.savez(f"ref_obs_{name}.npz", theta=theta_jax, one_run=np.asarray(one_run))
    _do_plots(one_run,name)
    return


def _do_plots(one_run,name):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    num_plots = min(one_run.shape[0], axes.size)
    print("one_run shape: ", one_run.shape)
    axes = axes.reshape(-1)
    for i in range(num_plots):
        ax = axes[i]
        img = np.asarray(one_run[i+1])
        im = ax.imshow(img, cmap='inferno', origin='lower', norm=LogNorm())
        ax.set_title(f"T = {(i+1)*0.5} yrs", fontsize=12)  # Now plots 1 through 11
        print("min and max of snapshot ", i+1, ": ", np.min(img), np.max(img))
        ax.axis('off')
    plt.tight_layout()
    # single colorbar for all subplots
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=1.0, aspect=12, pad=0.02)
    cbar.set_label('Counts', fontsize=14)
    plt.savefig(f"ref_{name}.png")
    plt.close()


if __name__ == "__main__":
    main()