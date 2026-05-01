import os
from autocvd import autocvd
# os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
autocvd(num_gpus = 1, interval = 1)

from sbi.diagnostics import run_tarp, check_tarp
from sbi.analysis.plot import plot_tarp
from sbi.diagnostics import run_sbc
from sbi.analysis.plot import sbc_rank_plot

# load_and_sample.py
import pickle
import torch
import jax 
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from astropy import units as u
from astropy import constants as c
from jf1uids import CodeUnits

from sbi.inference import NPE
from sbi.utils import BoxUniform
import torch.nn as nn
from sbi.neural_nets import posterior_nn

plt.rcParams.update({
    'axes.labelsize': 14,   # x and y labels
    'axes.titlesize': 16,   # plot title
    'xtick.labelsize': 12,  # x tick labels
    'ytick.labelsize': 12,  # y tick labels
    'legend.fontsize': 12,  # legend text
})


# posterior = inference.build_posterior()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

R_forced = 0.1
sep_in_au = 10  #upper-limit: 80, lower-limit: 5
length_temp = sep_in_au / 0.1  # before
mass_temp = 0.01 / 1  #1e-4
code_length = length_temp * u.au 
code_mass = mass_temp * u.M_sun
code_velocity = np.sqrt(c.G * code_mass / code_length).to(u.km/u.s)     # code_velocity = velocity_temp * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

mlr_lower = 8e-9 * u.M_sun / u.yr            # before in code units: [1e-13 / 1.309, 1e-8 / 1.309, 20, 90] 5.16e-9 M_sun/yr 5.16e-4 solMass/yr 800km/s 3600km/s
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
    high=torch.tensor([log_mlr_upper, log_mlr_upper, wind_vel_upper, wind_vel_upper, e_upper, cos_inc_upper, turbulence_strength_upper]),
    device=device
)    

save_path = os.path.join('trial20') #, 'reference_run2')


simulations = torch.load("sim_data/test_data_960_full.pt", map_location="cpu")
theta = simulations.get("theta")
x = simulations.get("x")
x = x[:,1:,:,:]

theta = theta.to(device)
x = x.to(device)

from scipy.stats import kstest

N_array = np.array([100, 500, 1000, 5000, 10000, 20000, 40000])   
results = {}

for N in N_array:
    with open(f"trial20/trained_posterior_adv_{N}_trial20.pkl", "rb") as f:
        posterior = pickle.load(f)
        # posterior._net.to(device)
    
    num_ppp = 500
    M = 500
    percentiles = np.zeros((4, num_ppp))
    posterior_mae = np.zeros((num_ppp, theta.shape[1]))
    posterior_rmse = np.zeros((num_ppp, theta.shape[1]))

    # Posterior spread metrics
    # Shape: (num_ppp, num_parameters)
    posterior_stds = np.zeros((num_ppp, theta.shape[1]))
    posterior_ci90_widths = np.zeros((num_ppp, theta.shape[1]))

    for i in range(num_ppp):
        samples = posterior.sample((M,), x=x[i]).cpu().numpy()
        true = theta[i].cpu().numpy().copy()

        posterior_mean = samples.mean(axis=0)
        err = posterior_mean - true

        posterior_mae[i] = np.abs(err)
        posterior_rmse[i] = np.sqrt(err**2)  # same as abs(err), but keeps intent clear

        for j in range(4):
            percentiles[j, i] = np.mean(samples[:, j] <= true[j])

        # spread of posterior for this x[i]
        posterior_stds[i] = np.std(samples, axis=0, ddof=1)

        # optional: 90% credible interval width
        q05 = np.quantile(samples, 0.05, axis=0)
        q95 = np.quantile(samples, 0.95, axis=0)
        posterior_ci90_widths[i] = q95 - q05

    # summarize over all num_ppp cases for this N
    mean_posterior_std = posterior_stds.mean(axis=0)
    mean_posterior_ci90_width = posterior_ci90_widths.mean(axis=0)
    mean_mae = posterior_mae.mean(axis=0)
    mean_rmse = posterior_rmse.mean(axis=0)

    
#     # this for computing ICE and mean PP dev

#     # for i in range(num_ppp):
#     #     samples = posterior.sample((M,), x=x[i]).cpu().numpy()      # optional sample with mcmc
#     #     true = theta[i].cpu().numpy().copy()
        
#     #     for j in range(4):
#     #         percentiles[j, i] = np.mean(samples[:, j] <= true[j])
    
#     # # Calculate PP-plot metrics
#     # pp_deviations = []
#     # for j in range(4):
#     #     p_sorted = np.sort(percentiles[j])
#     #     u = np.linspace(0, 1, num_ppp)
#     #     deviation = np.mean(np.abs(p_sorted - u))
#     #     pp_deviations.append(deviation)
    
#     # # Calculate ICE
#     # ecp, alpha = run_tarp(theta, x, posterior, references=None, 
#     #                       num_posterior_samples=500, use_batched_sampling=False)
#     # ice = (ecp - alpha).abs().mean().item()
    
#     # results[N] = {'ice': ice, 'pp_deviations': pp_deviations}
#     # print(f"N={N}: ICE={ice:.4f}, PP deviations={[f'{d:.4f}' for d in pp_deviations]}")


    results[N] = {
        'mean_posterior_std': mean_posterior_std,
        'mean_posterior_ci90_width': mean_posterior_ci90_width,
        'mean_mae': mean_mae,
        'mean_rmse': mean_rmse,
    }


np.savez("trial20/posterior_std.npz", 
         N_array=N_array, 
         results=results)

load_existing = True
if load_existing:
    data = np.load("trial20/posterior_std.npz", allow_pickle=True)
    N_array = data['N_array']
    results = data['results'].item()

# # Plot results (ICE/PP)
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# axes[0].plot(N_array, [results[N]['ice'] for N in N_array], marker='o')
# axes[0].set_xlabel('N samples')
# axes[0].set_ylabel('ICE')
# axes[0].set_title('ICE vs Training Samples')
# axes[0].grid()

# for j in range(4):
#     axes[1].plot(N_array, [results[N]['pp_deviations'][j] for N in N_array], marker='o', label=f'param {j}')
# axes[1].set_xlabel('N samples')
# axes[1].set_ylabel('Mean PP deviation')
# axes[1].set_title('PP-plot Deviation vs Training Samples')
# axes[1].legend()
# axes[1].grid()

# plt.tight_layout()
# plt.savefig(save_path + "/posterior_metrics_comparison.png", dpi=150)
# plt.close()


# # Plot results
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# axes[0].plot(N_array, [results[N]['ice'] for N in N_array], marker='o')
# axes[0].set_xlabel('N samples')
# axes[0].set_xscale('log')
# axes[0].set_ylabel('ICE')
# axes[0].set_title('ICE vs Training Samples')
# axes[0].grid()

# for j in range(4):
#     axes[1].plot(N_array, [results[N]['pp_deviations'][j] for N in N_array], marker='o', label=f'param {j}')
# axes[1].set_xlabel('N samples')
# axes[1].set_ylabel('Mean PP deviation')
# axes[1].set_xscale('log')
# axes[1].set_title('PP-plot Deviation vs Training Samples')
# axes[1].legend()
# axes[1].grid()

# plt.tight_layout()
# plt.savefig(save_path + "/posterior_metrics_comparison_b.png", dpi=150)
# plt.close()

param_scale = np.median(np.abs(theta.cpu().numpy()), axis=0)
normalized_std = [results[N]['mean_posterior_std'] for N in N_array] / param_scale
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

labels=[r"$\dot M_1$",r"$\dot M_2$",r"$v_{\inf,1}$",r"$v_{\inf,2}$", "e", "i", r"$\eta$"]

for j in range(theta.shape[1]):
    ax.plot(
        N_array,
        normalized_std[:, j],
        marker='o',
        label=f'{labels[j]}'
    )
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlabel('Training sample size (N)')
ax.set_ylabel(r'Mean posterior $1\sigma$ value (normalized)')
# ax.set_title('Posterior Std vs Training Samples')
ax.legend()
ax.grid()

plt.tight_layout()
plt.savefig(save_path + "/posterior_width_metrics.png", dpi=150)
plt.close()


param_scale = np.median(np.abs(theta.cpu().numpy()), axis=0)
normalized_mean_mae = [results[N]['mean_mae'] for N in N_array] / param_scale
normalized_mean_rsme = [results[N]['mean_rmse'] for N in N_array] / param_scale
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for j in range(theta.shape[1]):
    ax.plot(
        N_array,
        normalized_mean_rsme[:, j],
        marker='o',
        label=f'{labels[j]}'
    )
ax.set_xscale('log')
# ax.set_title('RMSE vs N')
ax.set_xlabel('Training sample size (N)')
ax.set_ylabel('RMSE (normalized)')
ax.legend()
ax.grid()

plt.tight_layout()
plt.savefig(save_path + "/posterior_error_per_param.png", dpi=150)
plt.close()