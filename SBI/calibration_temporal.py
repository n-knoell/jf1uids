import os
from autocvd import autocvd
# os.environ["CUDA_VISIBLE_DEVICES"] = "9" 
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
from scipy.stats import kstest


plt.rcParams.update({
    'axes.labelsize': 14,   # x and y labels
    'axes.titlesize': 16,   # plot title
    'xtick.labelsize': 12,  # x tick labels
    'ytick.labelsize': 12,  # y tick labels
    'legend.fontsize': 10,  # legend text
})

# posterior = inference.build_posterior()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path = os.path.join('trial20') #, 'reference_run2')

# with open("trained_posterior_combined.pkl", "rb") as f:
with open(save_path + "/trained_posterior_adv_40000_trial20.pkl", "rb") as f:
    posterior = pickle.load(f)
try:
    posterior._net.to(device)   
except Exception:
    pass


simulations = torch.load("sim_data/test_data_960_full.pt", map_location="cpu")
theta = simulations.get("theta")
x = simulations.get("x")
x = x[:,1:,:,:]

print("x shape: ", x.shape[0])
# print("x", x)

theta = theta.to(device)
x = x.to(device)

theta_200 = theta
x_200 = x

R_forced = 0.1
sep_in_au = 10  #upper-limit: 80, lower-limit: 5
length_temp = sep_in_au / 0.1  # before
mass_temp = 0.01 / 1  #1e-4
code_length = length_temp * u.au 
code_mass = mass_temp * u.M_sun
code_velocity = np.sqrt(c.G * code_mass / code_length).to(u.km/u.s)     # code_velocity = velocity_temp * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

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

prior = BoxUniform(
    low=torch.tensor([log_mlr_lower, log_mlr_lower, wind_vel_lower, wind_vel_lower, e_lower, cos_inc_lower, turbulence_strength_lower]),
    high=torch.tensor([log_mlr_upper, log_mlr_upper, wind_vel_upper, wind_vel_upper, e_upper, cos_inc_upper, turbulence_strength_upper]),
    device=device
)    

num_tarp_samples = 2000  

for i in [0,1,2,3]:
    print("min/max theta {i}: ", min(theta[:,i]), max(theta[:,i]))

from scipy.optimize import curve_fit


# linear model
def linear(x, a, b):
    return a * x + b

param1_predicted_mean = []
param2_predicted_mean = []
param3_predicted_mean = []
param4_predicted_mean = []
param5_predicted_mean = []
param6_predicted_mean = []
param7_predicted_mean = []

param1_predicted_std = []
param2_predicted_std = []
param3_predicted_std = []
param4_predicted_std = []
param5_predicted_std = []
param6_predicted_std = []
param7_predicted_std = []

param1_true = []
param2_true = []
param3_true = []
param4_true = []
param5_true = []
param6_true = []
param7_true = []


def v_code_to_real(v_code):
    return (v_code * code_units.code_velocity).to(u.km / u.s).value

def mlr_code_to_real(mlr_code):
    return (mlr_code * (code_units.code_mass / code_units.code_time)).to(u.M_sun / u.yr).value

m_conv_factor = (1 * (code_units.code_mass / code_units.code_time)).to(u.M_sun / u.yr).value

log2 = False    # whether to apply exp to the first two params for plotting (if they were log-sampled)
to_real_units = True  # whether to convert from code units to physical units for plotting

Number_of_datapoints = 200
first_index = 0


for i in range(first_index, first_index + Number_of_datapoints):  #theta.shape[0]):
    theta_predicted = posterior.sample((2000,), x=x[i]).cpu().numpy()
    theta_true = theta[i].cpu().numpy()

    if log2:
        theta_true[0] = np.exp(theta_true[0]) * m_conv_factor
        theta_true[1] = np.exp(theta_true[1]) * m_conv_factor
        theta_predicted[:, 0] = np.exp(theta_predicted[:, 0]) * m_conv_factor
        theta_predicted[:, 1] = np.exp(theta_predicted[:, 1]) * m_conv_factor

    if to_real_units:
        theta_true[0] = np.log10(m_conv_factor) + theta_true[0] * np.log10(np.e)  #mlr_code_to_real(theta_true[0])
        theta_true[1] = np.log10(m_conv_factor) + theta_true[1] * np.log10(np.e)  #mlr_code_to_real(theta_true[1])

        theta_true[2] = v_code_to_real(theta_true[2])
        theta_true[3] = v_code_to_real(theta_true[3])

        # theta_true[5] = np.arccos(theta_true[5]) 
        # theta_predicted[:, 5] = np.arccos(theta_predicted[:, 5])

        theta_predicted[:, 0] = np.log10(m_conv_factor) + theta_predicted[:, 0] * np.log10(np.e)  #mlr_code_to_real(theta_predicted[:, 0])
        theta_predicted[:, 1] = np.log10(m_conv_factor) + theta_predicted[:, 1] * np.log10(np.e)  #mlr_code_to_real(theta_predicted[:, 1])

        theta_predicted[:, 2] = v_code_to_real(theta_predicted[:, 2])
        theta_predicted[:, 3] = v_code_to_real(theta_predicted[:, 3])


    means = np.mean(theta_predicted, axis=0)
    stds  = np.std(theta_predicted, axis=0)

    param1_predicted_mean.append(means[0])
    param2_predicted_mean.append(means[1])
    param3_predicted_mean.append(means[2])
    param4_predicted_mean.append(means[3])
    param5_predicted_mean.append(means[4])
    param6_predicted_mean.append(means[5])
    param7_predicted_mean.append(means[6])

    param1_predicted_std.append(stds[0])
    param2_predicted_std.append(stds[1])
    param3_predicted_std.append(stds[2])
    param4_predicted_std.append(stds[3])
    param5_predicted_std.append(stds[4])
    param6_predicted_std.append(stds[5])
    param7_predicted_std.append(stds[6])

    param1_true.append(theta_true[0])
    param2_true.append(theta_true[1])
    param3_true.append(theta_true[2])
    param4_true.append(theta_true[3])
    param5_true.append(theta_true[4])
    param6_true.append(theta_true[5])
    param7_true.append(theta_true[6])

# convert to arrays
p_true = [
    np.array(param1_true),
    np.array(param2_true),
    np.array(param3_true),
    np.array(param4_true),
    np.array(param5_true),
    np.array(param6_true),
    np.array(param7_true),
]

p_mean = [
    np.array(param1_predicted_mean),
    np.array(param2_predicted_mean),
    np.array(param3_predicted_mean),
    np.array(param4_predicted_mean),
    np.array(param5_predicted_mean),
    np.array(param6_predicted_mean),
    np.array(param7_predicted_mean),
]

p_std = [
    np.array(param1_predicted_std),
    np.array(param2_predicted_std),
    np.array(param3_predicted_std),
    np.array(param4_predicted_std),
    np.array(param5_predicted_std),
    np.array(param6_predicted_std),
    np.array(param7_predicted_std),
]

# labels = [r"$\dot M_1$", r"$\dot M_2$", r"$v_{\infty,1}$", r"$v_{\infty,2}$"]
labels = [r"$\log \dot M_1$ [log(M$_{\odot}$/yr)]", r"$\log \dot M_2$ [log(M$_{\odot}$/yr)]", r"$v_{\infty,1}$ [km/s]", r"$v_{\infty,2}$ [km/s]", r"Eccentricity $e$", r"Inclination $\cos(i)$", r"Turbulence parameter $\eta$"]
raw_labels=[r"$\log \dot M_1$", r"$\log \dot M_2$", r"$v_{\infty,1}$", r"$v_{\infty,2}$", r"$e$", r"$\cos(i)$", r"$\eta$"]

fig, axes = plt.subplots(1, 7, figsize=(30, 4))
# fig, axes = plt.subplots(2, 4, figsize=(20, 10))
# axes = axes.flatten()[:7]

for ax, xdata, ydata, yerr, label, raw_label in zip(axes, p_true, p_mean, p_std, labels, raw_labels):
    # scatter with errorbars
    ax.errorbar(xdata, ydata, yerr=yerr, fmt='o', ms=3, capsize=2, alpha=0.7)

    # curve_fit with errors
    popt, pcov = curve_fit(linear, xdata, ydata, sigma=yerr, absolute_sigma=True)
    a, b = popt

    xfit = np.linspace(np.min(xdata), np.max(xdata), 200)
    yfit = linear(xfit, a, b)

    ax.plot(xfit, yfit, linewidth=1.5, label=f"fit y=ax+b") #label=f"fit: y={a:.3g}x+{b:.3g}")
    ax.plot(xfit, xfit, linestyle="--", linewidth=1.0, label="y=x")

    ax.set_xlabel(f"true {raw_label}")
    ax.set_ylabel(f"predicted {raw_label}")
    ax.set_title(f"{label}")
    ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig(save_path + "/params_full"+str(Number_of_datapoints)+"_TEST.png", dpi=250)
plt.show()

p_true_arr = np.stack(p_true, axis=0)
p_mean_arr = np.stack(p_mean, axis=0)
p_std_arr = np.stack(p_std, axis=0)

# np.savez_compressed(
#     os.path.join(save_path, "params_comparison.npz"),
#     p_true=p_true_arr,
#     p_mean=p_mean_arr,
#     p_std=p_std_arr,
# )

print("Params.png saved")

# --- Percentile-Percentile ---

num_ppp = x.shape[0]  #1000                    # number of held-out examples (matches the earlier loop)
M = 1000                         # posterior samples per example (matches earlier)
plot_num = 7
percentiles = np.zeros((plot_num, num_ppp))

raw_labels=[r"$\dot M_1$", r"$\dot M_2$", r"$v_{\infty,1}$ ", r"$v_{\infty,2}$", r"$e$", r"$i$", r"$\eta$"]

for i in range(num_ppp):
    # draw M posterior samples for the i-th observation
    samples = posterior.sample((M,), x=x[i]).cpu().numpy()   # shape (M,plot_num)
    true = theta[i].cpu().numpy().copy()                    # shape (plot_num,)

    # apply the same transform you used earlier (if using log for first two params)
    if log2:
        samples[:, 0] = np.exp(samples[:, 0])
        samples[:, 1] = np.exp(samples[:, 1])
        true[0] = np.exp(true[0])
        true[1] = np.exp(true[1])

    # percentile = fraction of posterior samples <= true value
    for j in range(plot_num):
        percentiles[j, i] = np.mean(samples[:, j] <= true[j])

# Plot PP-plots (sorted percentiles vs uniform)
fig, axes = plt.subplots(1, plot_num, figsize=(28, 4))
u = np.linspace(0, 1, num_ppp)

for j, ax in enumerate(axes):
    p_sorted = np.sort(percentiles[j])
    ax.plot(u, p_sorted, marker='o', ms=3, linestyle='-', label='empirical')
    ax.plot([0, 1], [0, 1], linestyle='--', label='ideal')
    ax.set_xlabel('Uniform quantile')
    ax.set_ylabel('Empirical percentile')
    ax.set_title(f'PP-plot for {raw_labels[j]}')
    ax.legend()

plt.tight_layout()
plt.savefig(save_path + "/pp_marginal_posterior_full_c.png", dpi=150)
plt.clf()
print("Saved pp_marginal_posterior_2log_opti.png")

# Kolmogorov-Smirnov test against Uniform(0,1) for each marginal
for j in range(plot_num):
    ks_stat, ks_p = kstest(percentiles[j, :], 'uniform')
    print(f"param {j}: KS p-value = {ks_p:.3f}")

# --- PIT (histogram of posterior percentiles) for each marginal parameter ---

# If `percentiles` not already computed by your PP code, compute it (all N examples)
try:
    percentiles  # 
except NameError:
    N = int(theta.shape[0])
    M = 2000
    percentiles = np.zeros((plot_num, N))
    for i in range(N):
        samples = posterior.sample((M,), x=x[i]).cpu().numpy()   # (M,plot_num)
        true = theta[i].cpu().numpy().copy()                     # (plot_num,)
        if log2:
            samples[:, 0] = np.exp(samples[:, 0])
            samples[:, 1] = np.exp(samples[:, 1])
            true[0] = np.exp(true[0])
            true[1] = np.exp(true[1])
        # use (rank+1)/(M+1) to avoid exact 0/1
        for j in range(plot_num):
            percentiles[j, i] = (np.sum(samples[:, j] <= true[j]) + 1) / (M + 1)

# Plot PIT histograms (expected flat=1)
bins = 10
fig, axes = plt.subplots(1, plot_num, figsize=(4*plot_num, 4))
for j, ax in enumerate(axes):
    ax.hist(percentiles[j], bins=bins, range=(0, 1), density=True, alpha=0.75, edgecolor='k')
    ax.axhline(1.0, linestyle='--', linewidth=1, label='uniform density')
    ax.set_xlim(0, 1)
    ax.set_xlabel('PIT value')
    ax.set_title(f'PIT param {j}')
    ax.legend(fontsize='small')

plt.tight_layout()
plt.savefig(save_path + "/pit_marginal_posterior_allN_full.png", dpi=150)
plt.clf()
print("Saved pit_marginal_posterior_allN_full.png")

# Optional numeric check: KS test vs Uniform(0,1)
for j in range(plot_num):
    ks_stat, ks_p = kstest(percentiles[j], 'uniform')
    print(f"param {j}: PIT KS p-value = {ks_p:.3f}")


theta_200 = theta
x_200 = x
# the tarp method returns the ECP values for a given set of alpha coverage levels.
ecp, alpha = run_tarp(
    theta_200,  #prior_samples,  #theta_200
    x_200, #prior_predictives,  #x_200
    posterior,
    references=None,  
    num_posterior_samples=1000,
    use_batched_sampling=False,  # `True` can give speed-ups, but can cause memory issues.
)

ice = (ecp - alpha).abs().mean().item()
print("ice: ", ice)

ecp = ecp.detach().cpu()
alpha = alpha.detach().cpu()

atc, ks_pval = check_tarp(ecp, alpha)
print(atc, "Should be close to 0")
print(ks_pval, "Should be larger than 0.05")

# number of held-out experiments used to compute ecp (should match the first dim of theta_200)
try:
    N = int(theta_200.shape[0])
except Exception:
    N = int(len(theta_200))

ecp_np = ecp.numpy()
alpha_np = alpha.numpy()
# standard error for binomial proportion p = ecp
se = np.sqrt((ecp_np * (1.0 - ecp_np)) / float(N))

# 95% Wald CI (normal approx). Clip to [0,1].
ci_low = np.clip(ecp_np - 1.96 * se, 0.0, 1.0)
ci_high = np.clip(ecp_np + 1.96 * se, 0.0, 1.0)

# statistic checks (keep using torch tensors as the functions expect)
atc, ks_pval = check_tarp(ecp, alpha)
print("atc:", atc, " (should be close to 0)")
print("ks_pval:", ks_pval, " (should be > 0.05 ideally)")

# plotting: use plot_tarp to draw base plot, then overlay shaded CI
f, ax = plot_tarp(ecp, alpha)  # plot_tarp accepts tensors (as used earlier)
ax.fill_between(alpha_np, ci_low, ci_high, alpha=0.25, label="95% CI")   #(binomial approx)
ax.legend(fontsize="small")

plt.tight_layout()
plt.savefig(save_path + "/tarp_plot_full.png", dpi=150)
plt.clf()
print("Saved tarp_plot_full.png")

raw_labels=[r"$\dot M_1$", r"$\dot M_2$", r"$v_{\infty,1}$ ", r"$v_{\infty,2}$", r"$e$", r"$i$", r"$\eta$"]

# run SBC: for each inference we draw 1000 posterior samples.
num_posterior_samples = 1_000
num_workers = 1
ranks, dap_samples = run_sbc(
    theta_200, #prior_samples,  #theta_200
    x_200, #prior_predictives,  #x_200
    posterior,
    num_posterior_samples=num_posterior_samples,
    num_workers=num_workers,
    use_batched_sampling=False, 
)

#SBC
f, ax = sbc_rank_plot(
    ranks=ranks,
    num_posterior_samples=num_posterior_samples,
    parameter_labels=raw_labels,
    # plot_type="hist",
    num_cols=3,
    figsize=(8, 8),
    num_bins=20,  # by passing None we use a heuristic for the number of bins.
)

plt.tight_layout()
plt.savefig(save_path + "/SBC_plot_full.png", dpi=250)
plt.clf()
