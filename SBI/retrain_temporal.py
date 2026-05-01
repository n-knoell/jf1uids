import os
from autocvd import autocvd
autocvd(num_gpus = 1, interval=1)

import pickle
import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from astropy import units as u
from astropy.constants import c
from jf1uids import CodeUnits

from sbi.inference import NPE
from sbi.utils import BoxUniform
import torch.nn as nn
from helper import DensityEmbeddingComplexTemporal, DensityEmbeddingPerFrameTemporal, DensityEmbeddingPerFrameTemporalAdv
from sbi.neural_nets import posterior_nn

from astropy import units as u
import astropy.constants as c
import jax
import jax.numpy as jnp
from jax import random as jrandom

from Halpha import add_detector_noise_jax


def add_fresh_noise(x_clean_torch, seed, chunk=500):
    """
    Draw a fresh detector-noise realisation for a batch of clean photon-count
    snapshots. 
    x_clean_torch : torch.Tensor, shape (N, T, H, W)
    seed : int
    chunk : int
        How many samples to push through JAX at once
    Returns torch.Tensor, shape (N, T, H, W)
        Noisy photon counts, same dtype/device as the input.
    """
    original_device = x_clean_torch.device
    original_dtype = x_clean_torch.dtype

    x_np = x_clean_torch.detach().cpu().numpy()
    N, T, H, W = x_np.shape

    def _noise_one(img, k):
        noisy_img, _ = add_detector_noise_jax(img, key=k)
        return jnp.clip(noisy_img, a_min=1.0, a_max=None)

    _noise_batched = jax.vmap(jax.vmap(_noise_one))  # (N, T, H, W), (N, T, 2)

    key = jrandom.PRNGKey(seed)
    out_pieces = []
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        sub = jnp.asarray(x_np[start:end])
        n_sub = end - start
        key, subkey = jrandom.split(key)
        sub_keys = jrandom.split(subkey, n_sub * T).reshape(n_sub, T, 2)
        out_pieces.append(np.asarray(_noise_batched(sub, sub_keys)))

    x_noisy_np = np.concatenate(out_pieces, axis=0)
    return torch.from_numpy(x_noisy_np).to(device=original_device, dtype=original_dtype)

##load simulations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

simulations = torch.load("simulations_clean_TEST.pt", map_location="cpu")  # <- point this at the CLEAN combined file
theta = simulations.get("theta")
x_clean = simulations.get("x")
x_clean = x_clean[:, 1:, :, :]
theta = theta.to(device)
# and only the noised batch is moved to `device`.

print("x_clean.shape:", x_clean.shape)
print("theta.shape:", theta.shape)

def make_embedding_from_trial_temporal( 
    num_conv_blocks,
    fc_dim,
    base_channels,
    temporal_pool_output,
    temporal_num_layers,
    temporal_stride_first,
    input_shape=(10,64,64)
    ):
    num_conv_blocks = num_conv_blocks
    fc_dim           = fc_dim
    base_channels    = base_channels   # small channel choices
    temporal_pool_output   = temporal_pool_output  # default 1
    temporal_num_layers    = temporal_num_layers   # default 2
    temporal_stride_first  = temporal_stride_first # default 1
    # base_channels = 32
    dropout = 0.0
    kernel_size = 3
    use_batchnorm = False
    activation = 'relu'

    embedding = DensityEmbeddingPerFrameTemporalAdv(
        input_shape=input_shape,
        num_conv_blocks=num_conv_blocks,
        base_channels=base_channels,
        kernel_size=kernel_size,
        use_batchnorm=use_batchnorm,
        dropout=dropout,
        fc_dim=fc_dim,
        activation=activation,
        temporal_pool_output=temporal_pool_output,
        temporal_num_layers=temporal_num_layers,
        temporal_stride_first=temporal_stride_first,
    )
    return embedding

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

#best combine: trial 14
# params={'num_conv_blocks': 3, 'fc_dim': 64, 'base_channels': 16, 
#     'temporal_pool_output': 1, 'temporal_num_layers': 1, 'temporal_stride_first': 2,
#      'hidden_features': 30, 'num_transforms': 20}

# best TARP: trial 20
# params: {'num_conv_blocks': 3, 'fc_dim': 128, 'base_channels': 32,
#      'temporal_pool_output': 2, 'temporal_num_layers': 2, 'temporal_stride_first': 1,
#       'hidden_features': 7, 'num_transforms': 20}

embedding_net = make_embedding_from_trial_temporal(
    num_conv_blocks = 3,
    fc_dim = 128,
    base_channels = 32,
    temporal_pool_output = 2,
    temporal_num_layers = 2,
    temporal_stride_first = 1
)

density_estimator_builder = posterior_nn(
    model="nsf",
    hidden_features=7,
    num_transforms=20,
    embedding_net=embedding_net
)

# inference = NPE(prior=prior, density_estimator=density_estimator_builder, device=device)

# results for different N
# N_array = np.array([20000])  

# for i in N_array:
#     print(f"Training with {i} simulations...")
#     inference = NPE(prior=prior, density_estimator=density_estimator_builder, device=device)
#     indices = torch.randperm(theta.shape[0])[:i]
#     theta_sample = theta[indices]

#     # ---- fresh noise realisation for THIS training run ---------------------
#     # A new seed is drawn every iteration so that each retrain sees different
#     # detector noise on top of the same underlying (clean) simulations.
#     noise_seed = int(np.random.randint(0, 2**31 - 1))
#     print(f"  noise seed: {noise_seed}")
#     x_sample_clean = x_clean[indices.cpu()]                 # CPU indexing
#     x_sample = add_fresh_noise(x_sample_clean, seed=noise_seed).to(device)
#     # ------------------------------------------------------------------------

#     density_estimator = inference.append_simulations(theta_sample, x_sample).train()
#     posterior = inference.build_posterior(density_estimator)
#     # with open(f"trial20/trained_posterior_adv_{i}_trial20.pkl", "wb") as f:
#     with open(f"trained_posterior_adv_{i}_TEST.pkl", "wb") as f:
#         pickle.dump(posterior, f)
#     print(f"Saved posterior -> trained_posterior_{i}.pkl")

Num_sims = 20
# Training with noise resampled every few epochs.


epochs_per_noise_round = 5      # resample noise every x epochs
max_noise_rounds       = 60     
stop_after_epochs      = 20     # early-stopping

print(f"Training with {Num_sims} simulations, "
      f"resampling noise every {epochs_per_noise_round} epochs "
      f"(cap: {max_noise_rounds * epochs_per_noise_round} total epochs)...")

inference = NPE(prior=prior, density_estimator=density_estimator_builder, device=device)
indices = torch.randperm(theta.shape[0])[:Num_sims]
theta_sample = theta[indices]

# Keep the clean slice on CPU
x_sample_clean = x_clean[indices.cpu()]

seed_0 = int(np.random.randint(0, 2**31 - 1))
print(f"[round 0] initial noise seed: {seed_0}")
x_sample_noisy = add_fresh_noise(x_sample_clean, seed=seed_0).to(device)
inference.append_simulations(theta_sample, x_sample_noisy)

density_estimator = None
for round_i in range(max_noise_rounds):
    if round_i > 0:
        # Swap in a fresh noise realisation of the SAME simulations.
        # NOTE: this mutates sbi's internal round-wise data buffer
        # (`_x_roundwise`). The attribute name has been stable across
        # sbi 0.22-0.25; if your sbi version renames it, adjust here.
        seed = int(np.random.randint(0, 2**31 - 1))
        x_new = add_fresh_noise(x_sample_clean, seed=seed).to(device)
        inference._x_roundwise[-1] = x_new
        del x_new
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[round {round_i}] noise seed: {seed}")

    # max_num_epochs is an ABSOLUTE cap in sbi 
    target_epochs = (round_i + 1) * epochs_per_noise_round
    density_estimator = inference.train(
        max_num_epochs    = target_epochs,
        stop_after_epochs = stop_after_epochs,
        resume_training   = (round_i > 0),
    )

    # early-stopping?
    no_improvement = getattr(inference, "_epochs_since_last_improvement", 0)
    if no_improvement >= stop_after_epochs:
        print(f"  -> converged after round {round_i + 1} "
              f"({no_improvement} epochs without val-loss improvement). Stopping.")
        break
else:
    print(f"  -> hit max_noise_rounds={max_noise_rounds} without convergence.")

posterior = inference.build_posterior(density_estimator)
with open(f"trained_posterior_adv_{Num_sims}_TEST.pkl", "wb") as f:
    pickle.dump(posterior, f)
print(f"Saved posterior -> trained_posterior_adv_{Num_sims}_TEST.pkl")


# #### single retraining with on all simulations
# density_estimator = inference.append_simulations(theta, x).train()
#     # show_train_summary=True,
#     # training_batch_size=1000,
# # )

# posterior = inference.build_posterior(density_estimator)

# with open("trained_posterior_adv_trial20.pkl", "wb") as f:
#     pickle.dump(posterior, f)

# # # Optionally also save the whole inference object:
# # with open("inference_object_combined.pkl", "wb") as f:
# #     pickle.dump(inference, f)

# print("Saved posterior -> trained_posterior_combined.pkl")