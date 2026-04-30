import os
from autocvd import autocvd
autocvd(num_gpus = 1)

# numerics
import jax
import jax.numpy as jnp
import numpy as np

# timing
from timeit import default_timer as timer
from jf1uids.option_classes.simulation_config import FORWARDS, HLL, VARAXIS, XAXIS, YAXIS, ZAXIS
from jf1uids.option_classes.simulation_config import finalize_config

from jax.sharding import PartitionSpec as P, NamedSharding
import torch 
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

# units
from jf1uids import CodeUnits
from astropy import units as u
import astropy.constants as c

from matplotlib.lines import Line2D

# turbulence
from jf1uids.initial_condition_generation.turb import create_turb_field
from jf1uids.option_classes.simulation_config import FORWARDS
from jf1uids.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD, OSHER, 
    PERIODIC_BOUNDARY, OPEN_BOUNDARY, REFLECTIVE_BOUNDARY, 
    BoundarySettings, BoundarySettings1D
)

plt.rcParams.update({
    'axes.labelsize': 15,   # x and y labels
    'axes.titlesize': 15,   # plot title
    # 'xtick.labelsize': 12,  # x tick labels
    # 'ytick.labelsize': 12,  # y tick labels
    'legend.fontsize': 10,  # legend text
})

jax.config.update("jax_enable_x64", True)

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
num_timesteps = 1600
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
    # time_integrator = RK2_SSP,
    time_integrator = MUSCL,
    cooling_config = CoolingConfig(
        cooling = False,
    ),
    wind_config = WindConfig(
        stellar_wind = stellar_wind,
        # num_injection_cells = 25,
        num_injection_cells = 2,
        trace_wind_density = False,
        real_wind_params = False,
    ),
    fixed_timestep = fixed_timestep,
    differentiation_mode = FORWARDS,
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

C_CFL = 0.4
R_forced = 0.1
sep_in_au = 10  #upper-limit: 80, lower-limit: 5
length_temp = sep_in_au / 0.1  # before
mass_temp = 0.01 / 1  #1e-4
code_length = length_temp * u.au 
code_mass = mass_temp * u.M_sun
code_velocity = np.sqrt(c.G * code_mass / code_length).to(u.km/u.s)     # code_velocity = velocity_temp * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

a = box_size / 3.8  ## always 0.1   # semi-major axis

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

import numpy as np
from functools import partial
from time import time
import matplotlib.pyplot as plt
from jax.scipy.linalg import cho_solve, cho_factor


def physical_summary(x):
    """
    Physics-motivated handcrafted summary for CWB Hα time series.
    Input:  x of shape [T, H, W] - photon counts
    Output: 1D jnp.array of length 6T + 2  (= 62 for T=10)
    
    Physical interpretation per component:
      log(total flux)[T]       → mass-loss rates Ṁ (jHα ∝ n² ∝ Ṁ²)
      log(peak flux)[T]        → stagnation-point brightness / Ṁ ratio
      centroid cx, cy [2T]     → orbital motion → e, i, phase
      spread σx, σy [2T]       → bow-shock opening angle → v∞
      flux_std/mean [1]        → periastron variability → e
      peak_ratio [1]           → periastron brightening → e
    """
    x = jnp.asarray(x).astype(jnp.float64)
    T, H, W = x.shape
    eps = 1e-12

    yy, xx = jnp.meshgrid(jnp.arange(H, dtype=jnp.float64),
                          jnp.arange(W, dtype=jnp.float64), indexing='ij')

    # per-frame statistics
    total = x.sum(axis=(1, 2))                                   # [T]
    peak  = x.max(axis=(1, 2))                                   # [T]
    denom = total + eps

    # brightness-weighted centroid (orbital motion signature)
    cx = (x * xx).sum(axis=(1, 2)) / denom                       # [T]
    cy = (x * yy).sum(axis=(1, 2)) / denom                       # [T]

    # brightness-weighted second moments (bow-shock extent)
    vx = ((xx - cx[:, None, None])**2 * x).sum(axis=(1, 2)) / denom
    vy = ((yy - cy[:, None, None])**2 * x).sum(axis=(1, 2)) / denom
    sig_x = jnp.sqrt(vx)                                         # [T]
    sig_y = jnp.sqrt(vy)                                         # [T]

    # global temporal variability (eccentricity signature)
    flux_std_over_mean = total.std() / (total.mean() + eps)
    peak_ratio = peak.max() / (peak.min() + eps)

    return jnp.concatenate([
        jnp.log10(total + eps),
        jnp.log10(peak + eps),
        cx, cy,
        sig_x, sig_y,
        jnp.array([flux_std_over_mean, peak_ratio])
    ])

# Distance between simulated summary and observed summary
from functools import partial

def compute_summary_norm(sim_array, summary_fn, max_sims=1000):
    """Precompute per-component mean/std from a batch of simulations."""
    n_use = min(max_sims, len(sim_array))
    summs = [np.asarray(summary_fn(sim_array[i])) for i in range(n_use)]
    summs = np.stack(summs, axis=0)
    mean = summs.mean(axis=0)
    std  = summs.std(axis=0) + 1e-8  # avoid div-by-zero
    return jnp.asarray(mean), jnp.asarray(std)


def standardized_l2(summ_sim, summ_obs, mean, std):
    """L2 distance in standardized (z-scored) summary space."""
    s_sim = (summ_sim - mean) / std
    s_obs = (summ_obs - mean) / std
    return jnp.sqrt(jnp.mean((s_sim - s_obs) ** 2))


def weighted_standardized_l2(summ_sim, summ_obs, mean, std, weights):
    s_sim = (summ_sim - mean) / std
    s_obs = (summ_obs - mean) / std
    return jnp.sqrt(jnp.mean(weights * (s_sim - s_obs) ** 2))


# ---------- ABC-Rejection ----------
def abc_rejection(observed_summary,
                  prior_ranges,
                  simulations = None,
                  num_samples=100,
                  epsilon=None,
                  batch_size=4,
                  rng_seed=0,
                  summary_fn=physical_summary,
                  distance_fn=standardized_l2,
                  simulator_fn=None,
                  save_path='ABC/abc_rejection.npz'):
    """
    Run ABC rejection sampling.
    - prior_ranges: list of (low,high) per parameter (same units as simulator expects)
    - num_samples: how many accepted posterior samples to collect
    - epsilon: acceptance threshold. If None will be set adaptively using pilot sims
    - batch_size: sample proposals per loop (helps amortize python loop overhead)
    """
    key = jax.random.PRNGKey(rng_seed)
    accepted = []
    dists = []

    if simulations is not None:
        # assume torch tensors (as in your main), but handle numpy too
        thetas_t = simulations.get("theta")
        sims_t = simulations.get("x")
        sims_t = sims_t[:, 1:, :, :]     # drop T=0 snapshot (initial condition)
        # safe conversion
        all_thetas = thetas_t.cpu().numpy() if hasattr(thetas_t, "cpu") else np.array(thetas_t)
        all_sims = sims_t.cpu().numpy() if hasattr(sims_t, "cpu") else np.array(sims_t)
        n_precomputed = all_thetas.shape[0]
    else:
        n_precomputed = 0

    if epsilon is None:
        pilot_n = 500  # you can change default
        k1, key = jax.random.split(key)
        if simulations is None:
            thetas_pilot = np.array(sample_prior_uniform(k1, pilot_n, prior_ranges))
        else:
            # use as many as available up to pilot_n
            use_n = min(pilot_n, n_precomputed)
            thetas_pilot = all_thetas[:use_n]
        pilot_d = []
        for idx, th in enumerate(thetas_pilot):
            if simulations is None:
                s = simulator_fn(th, summary_fn=summary_fn)
                summ_s = summary_fn(s)
            else:
                s = all_sims[idx]
                summ_s = summary_fn(s)
            pilot_d.append(float(distance_fn(summ_s, observed_summary)))
        upper_lim = 0.05
        epsilon = np.quantile(np.array(pilot_d), upper_lim)
        print(f"[ABC] Pilot epsilon set to {epsilon:.3e} ({upper_lim*100:.2f}% quantile)")
    attempts = 0
    t0 = time()

    sim_index = 0  # pointer into precomputed arrays when simulations is given

    while len(accepted) < num_samples:
        attempts += 1
        k1, key = jax.random.split(key)
        # draw proposals
        if simulations is None:
            thetas = np.array(sample_prior_uniform(k1, batch_size, prior_ranges))
        else:
            # slice next `batch_size` entries (stop if exhausted)
            if sim_index >= n_precomputed:
                break
                # raise RuntimeError("[ABC] Out of precomputed simulations while still collecting posterior samples.")
            end_idx = min(sim_index + batch_size, n_precomputed)
            thetas = all_thetas[sim_index:end_idx]
        # evaluate each proposal
        for idx, th in enumerate(thetas):
            if simulations is None:
                s = simulator_fn(th, summary_fn=summary_fn)
                summ_s = summary_fn(s)
            else:
                s = all_sims[sim_index + idx]
                summ_s = summary_fn(s)
            dist = float(distance_fn(summ_s, observed_summary))
            if dist <= epsilon:
                accepted.append(th)
                dists.append(dist)
                if len(accepted) >= num_samples:
                    break
        # advance pointer when using precomputed sims
        if simulations is not None:
            sim_index += len(thetas)

        if attempts % 10 == 0:
            # compute how many proposals we've actually generated so far:
            proposals_done = (attempts * batch_size) if simulations is None else sim_index
            print(f"[ABC] Attempts (loops): {attempts}, proposals processed: {proposals_done}, accepted {len(accepted)}/{num_samples}")

    t1 = time()
    print(f"[ABC] Finished: accepted {len(accepted)} samples in {t1-t0:.1f}s (loops: {attempts})")
    acc = np.array(accepted)
    dists = np.array(dists)
    np.savez_compressed(save_path, posterior=acc, distances=dists, epsilon=epsilon)
    return acc, dists, epsilon

def convert_to_phys(c_unit, param_idx):
    if param_idx <= 1:   # log mass-loss rates
        return np.exp(c_unit) * (1 * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value
    elif param_idx <= 3:  # wind velocities
        return (c_unit * code_units.code_velocity).to(u.km / u.s).value
    else:                 # e, cos(i), η — dimensionless
        return c_unit

# ---------- Posterior diagnostics 
def posterior_68_ci_width(posterior, prior_ranges):
    """
    Compute the 68% credible-interval width per parameter, in the SAME
    display units used in the LaTeX Table 4:

        idx 0, 1 (mass-loss rates):  log10(M_sun / yr)
        idx 2, 3 (wind velocities):  km / s
        idx 4    (eccentricity):     dimensionless
        idx 5    (cos i):            dimensionless
        idx 6    (eta):              dimensionless

    Returns a list of dicts (one per parameter):
        {'label', 'width_post', 'width_prior', 'ratio'}
    where ratio = width_post / width_prior. Values close to 1 indicate
    "essentially the prior" -> mark as "prior" in the table.
    """
    labels = ['log M_dot_1', 'log M_dot_2',
              'v_inf_1', 'v_inf_2',
              'e', 'cos i', 'eta']
    out = []
    for i in range(posterior.shape[1]):
        # ---- posterior in display units ----
        s = convert_to_phys(posterior[:, i], i)
        if i <= 1:                       # log10 for mass-loss rates
            s = np.log10(s)
        q16, q84 = np.percentile(s, [16, 84])
        width_post = q84 - q16

        # ---- prior in display units (uniform -> 68% CI = 0.68 * range) ----
        plo = convert_to_phys(prior_ranges[i][0], i)
        phi = convert_to_phys(prior_ranges[i][1], i)
        if i <= 1:
            plo, phi = np.log10(plo), np.log10(phi)
        width_prior = 0.68 * (phi - plo)

        out.append({
            'label':       labels[i],
            'width_post':  width_post,
            'width_prior': width_prior,
            'ratio':       width_post / width_prior,
        })
    return out


def report_widths(posterior, prior_ranges, name=""):
    rows = posterior_68_ci_width(posterior, prior_ranges)

    print(f"\n=== ABC posterior 68% CI widths ({name}) ===")
    print(f"{'parameter':<14} {'width(post)':>12} {'width(prior)':>13} "
          f"{'post/prior':>11}  note")
    print("-" * 64)
    for r in rows:
        flag = "≈ prior" if r['ratio'] > 0.85 else ""
        print(f"{r['label']:<14} {r['width_post']:>12.4f} "
              f"{r['width_prior']:>13.4f} {r['ratio']:>11.3f}  {flag}")

    print(f"\n--- snippet ({name}) ---")
    fmt = lambda x: f"{x:.2f}"
    for r in rows:
        if r['ratio'] > 0.85:
            print(f"  {r['label']:<14} & --- & prior     & $\\gg 1$ \\\\")
        else:
            print(f"  {r['label']:<14} & --- & {fmt(r['width_post']):>8} & --- \\\\")
    return rows

from scipy.stats import pearsonr

def plot_marginals(samples, labels=None, savefile=None, bins=50, prior_ranges=None):
    """
    Basic pairwise marginal plotting for the posterior samples (numpy array shape [N, D]).
    """
    use_log = True
    N, D = samples.shape
    fig, axs = plt.subplots(D, D, figsize=(17.5, 17.5))
    # Convert prior bounds to physical units
    prior_bounds_phys = None
    if prior_ranges is not None:
        prior_bounds_phys = [(convert_to_phys(p[0], i), convert_to_phys(p[1], i)) 
                             for i, p in enumerate(prior_ranges)]
    n_bins = bins
    for i in range(D):
        for j in range(D):
            ax = axs[i, j]

            if i == j:
                x = samples[:, i]
                x = convert_to_phys(samples[:, i], i)
                if i < 2 and use_log:
                    bin_edges = np.logspace(np.log10(x.min()),
                                             np.log10(x.max()),
                                             n_bins)
                    ax.hist(x, bins=bin_edges)
                else:
                    ax.hist(x, bins=n_bins)
                ax.axvline(np.median(x), color='r', linestyle='--', label='median')
                ax.axvline(true_values_phys[i], color='g', linestyle='-', label='true value')
                if i < 2 and use_log:
                    ax.set_xscale('log')
                # Set xlim from prior bounds
                if prior_bounds_phys is not None:
                    ax.set_xlim(prior_bounds_phys[i])
                #visualize the 1 sigma confidence interval
                #ci_lower = np.percentile(samples[:, i], 34.1)
                #ci_upper = np.percentile(samples[:, i], 65.9)
                #ax.axvline(ci_lower, color='b', linestyle=':', label=r'$1\sigma$ CI')
                #ax.axvline(ci_upper, color='b', linestyle=':')
                # ax.legend()
            elif i > j:
                x = samples[:, i]
                x = convert_to_phys(samples[:, i], i)
                y = samples[:, j]
                y = convert_to_phys(samples[:, j], j)
                r, p_value = pearsonr(y, x)
                ax.scatter(y, x, s=2)
                ax.set_title(r'$\rho_{X,Y}$='+str(np.round(r,2)), fontsize=10)
                # Set xlim and ylim from prior bounds
                if prior_bounds_phys is not None:
                    ax.set_xlim(prior_bounds_phys[j])
                    ax.set_ylim(prior_bounds_phys[i])
                if i < 2 or j < 2 and use_log:
                    ax.set_xscale('log')
            else:
                ax.axis('off')
            # Add labels on outer edges
            if labels:
                # x-labels on bottom row
                if i == D - 1:
                    axs[i, j].set_xlabel(labels[j])
                # y-labels on left column
                if j == 0 and i != 0:  # skip (0,0) to avoid overlap with diagonal
                    axs[i, j].set_ylabel(labels[i])
    # Create single legend for the entire figure
    legend_elements = [Line2D([0], [0], color='r', linestyle='--', lw=2, label='median'),
                       Line2D([0], [0], color='g', linestyle='-', lw=2, label='true value')]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.tight_layout()
    if savefile:
        fig.savefig(savefile, dpi = 300)
    return fig

# usage
if __name__ == "__main__":
    save_path = os.path.join('ABC')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = "diff"
    data = np.load(f"ref_obs_{name}.npz", allow_pickle=True)
    one_run = torch.as_tensor(data["one_run"], device=device)
    theta_jax = torch.as_tensor(data["theta"], device=device)
    theta_o = theta_jax
    x_o = one_run[1:,:,:] #becaue we dont need the first snapshot at T=0, which is just the initial condition
    true_values = theta_o.cpu().numpy().flatten()
    true_values_phys = []
    for idx, val in enumerate(true_values):
        true_values_phys.append(convert_to_phys(val, idx))
    print("True values (physical units): ", true_values_phys)

    # Compute observed summary
    obs_summary = physical_summary(x_o)

    simulations = torch.load("sim_data/full_comb_40k.pt", map_location="cpu")
    theta = simulations.get("theta")
    x = simulations.get("x")
    x = x[:, 1:, :, :]     
    all_sims = x.numpy()                                              # ← needed for norm

    obs_summary = physical_summary(x_o)
    stat_mean, stat_std = compute_summary_norm(all_sims, physical_summary, max_sims=1000)
    distance_fn_std = partial(standardized_l2, mean=stat_mean, std=stat_std)

    mlr_lower = 8e-9 * u.M_sun / u.yr         
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

    prior_ranges = [
        (log_mlr_lower, log_mlr_upper),   # mdot1 (code units)  [1e-13 / 1.309, 1e-8 / 1.309, 20, 90]
        (log_mlr_lower, log_mlr_upper),    # mdot2 (code units)
        (wind_vel_lower, wind_vel_upper),     # v_inf1 (code units)
        (wind_vel_lower, wind_vel_upper),     # v_inf2
        (e_lower, e_upper),         # e (code units)
        (cos_inc_lower, cos_inc_upper),  # cos(i) (code units)
        (turbulence_strength_lower, turbulence_strength_upper)  # η (code units)
    ]

    # ABC-Rejection quick run
    posterior, dists, eps = abc_rejection(
        observed_summary=obs_summary,
        prior_ranges=prior_ranges,
        simulations = simulations,
        num_samples=2000,  
        batch_size=4,
        rng_seed=1,
        summary_fn=physical_summary,
        distance_fn=distance_fn_std,
        save_path=save_path+'/abc_rejection_'+str(name)+'.npz'
    )
    print("ABC-Rejection sample shape:", posterior.shape)

    np.savez_compressed(save_path+'/posterior_'+str(name)+'.npz', arr=posterior)
    np.savez_compressed(save_path+'/dists_'+str(name)+'.npz', arr=dists)
    np.savez_compressed(save_path+'/eps_'+str(name)+'.npz', arr=eps)    
    
    plot_marginals(posterior, labels=[r'$\dot M_1$',r'$\dot M_2$',r'$v_{inf1}$',r'$v_{inf2}$', r'$e$', r'$\cos(i)$', r'$\eta$'], savefile=save_path+'/abc_rejection_'+str(name)+'.png', prior_ranges=prior_ranges)

    print(report_widths(posterior, prior_ranges, name=name))