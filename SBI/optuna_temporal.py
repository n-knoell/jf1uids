import os
from autocvd import autocvd
autocvd(num_gpus = 1, interval = 1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

import numpy as np
from scipy.stats import norm

import jax.numpy as jnp

import sbi.utils as utils
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from sbi.inference import simulate_for_sbi, NPE
from sbi.neural_nets import posterior_nn
from sbi.diagnostics import run_tarp, check_tarp

import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import time as t
import pickle
from tqdm import tqdm
from tarp import get_tarp_coverage
import tarp

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

import torch.nn as nn
from sbi.utils import BoxUniform
import torch
# import torch.nn.functional as F
from helper import DensityEmbeddingComplexTemporal, DensityEmbeddingPerFrameTemporal, DensityEmbeddingPerFrameTemporalAdv

from jf1uids import CodeUnits
from astropy import units as u
import astropy.constants as c

import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException


def compute_log_prob_safe(posterior, theta, x, chunk_size=100):
    """Try full batch -> chunked -> per-sample fallback.
    Returns a 1D tensor of log-prob values (length == theta.shape[0])."""
    # fast: try the full batch first
    try:
        lp = posterior.log_prob_batched(theta, x=x, norm_posterior=False)
        return lp.reshape(-1)
    except Exception as full_err:
        # fallback: chunked evaluation
        lps = []
        n = theta.shape[0]
        for i in range(0, n, chunk_size):
            th = theta[i : i + chunk_size]
            xx = x[i : i + chunk_size]
            try:
                lp_chunk = posterior.log_prob_batched(th, x=xx, norm_posterior=False)
                lps.append(lp_chunk.reshape(-1))
            except AssertionError:
                for j in range(th.shape[0]):
                    th_j = th[j : j + 1]
                    xx_j = xx[j : j + 1]
                    try:
                        lp_j = posterior.log_prob(th_j, x=xx_j,  norm_posterior=False)
                    except Exception:
                            # last resort: try log_prob_batched on single sample
                        lp_j = posterior.log_prob_batched(th_j, x=xx_j,  norm_posterior=False)
                    lps.append(lp_j.reshape(-1))
        return torch.cat(lps, dim=0)



def make_embedding_from_trial_temporal(trial, input_shape=(10,64,64)):
    num_conv_blocks = trial.suggest_categorical("num_conv_blocks", [2, 3])
    fc_dim           = trial.suggest_categorical("fc_dim", [64, 128, 256])   
    base_channels    = trial.suggest_categorical("base_channels", [16, 32])          # small channel choices
    # dropout          = trial.suggest_categorical("dropout", [0.0, 0.12])           # light regularization

    temporal_pool_output   = trial.suggest_categorical("temporal_pool_output", [1, 2, 4])   # default 1
    temporal_num_layers    = trial.suggest_int("temporal_num_layers", 1, 3)               # default 2
    temporal_stride_first  = trial.suggest_categorical("temporal_stride_first", [1, 2])  # default 1

    # New minimal temporal hyperparameters sampled from trial:
    # temporal_ks = trial.suggest_categorical("temporal_ks", [1, 3])      # 1 = no temporal context; 3 = local temporal context
    # temporal_stride = trial.suggest_categorical("temporal_stride", [1, 2])  # 1 = preserve time, 2 = downsample time

    # Keep some fixed choices for stability, as you had before
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

def objective(trial):

    embedding_net = make_embedding_from_trial_temporal(trial) ## complex

    # model_type = trial.suggest_categorical("model", ['nsf', 'maf', 'mdn'])
    hidden_features = trial.suggest_int("hidden_features", 4, 30, log=False)
    num_transforms  = trial.suggest_int("num_transforms", 3, 25, log=False)

    # 2) build density estimator builder using embedding
    # density_estimator_build_fun = make_density_estimator_builder(trial, embedding_net)
    density_estimator_build_fun = posterior_nn(
        # model=model_type,
        model="nsf",
        embedding_net=embedding_net,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
    )
    # build inference object 
    inference = NPE(prior=prior, density_estimator=density_estimator_build_fun, show_progress_bars=True, device="cuda")
    print("Inference object built")
    # Training hyperparams from trial
    # lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    # batch_size = trial.suggest_categorical("batch_size", [128, 512, 1024, 4096])

    # simple training call
    density_estimator = inference.append_simulations(theta, x).train(
        show_train_summary=True,
        training_batch_size=5000,
    )
    print("density estimator trained")
    posterior = inference.build_posterior(density_estimator)
    
    print("posterior built")

    # # NLL
    lp = compute_log_prob_safe(posterior, theta_test, x_test, chunk_size=80)
    nll_test = -lp.mean()
    print("NLL computed")

    num_samples = 500
    theta_t_cut = theta_test[:num_samples]
    x_t_cut = x_test[:num_samples] 
   
    ecp, alpha = run_tarp(
        theta_t_cut,
        x_t_cut,
        posterior,
        num_posterior_samples=500,
        use_batched_sampling=True,
    )
    print("TARP run")
    ice = (ecp - alpha).abs().mean().item()

    return nll_test, ice
    # return ice

R_forced = 0.1
sep_in_au = 10  #upper-limit: 80, lower-limit: 5
length_temp = sep_in_au / 0.1  # before
mass_temp = 0.01 / 1  #1e-4
code_length = length_temp * u.au 
code_mass = mass_temp * u.M_sun
code_velocity = np.sqrt(c.G * code_mass / code_length).to(u.km/u.s)     # code_velocity = velocity_temp * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

if __name__ == '__main__':
    # # sanity check for embedding net
    # net = DensityEmbeddingPerFrameTemporalAdv(input_shape=(10, 64, 64),     # (T, H, W)
    #     num_conv_blocks=3,
    #     base_channels=16,
    #     kernel_size=3,
    #     use_batchnorm=True,
    #     dropout=0.0,
    #     fc_dim=32,
    #     activation='relu',
    #     temporal_pool_output=2,
    #     temporal_num_layers=1,
    #     temporal_stride_first=1,
    #     # temporal_ks=3,         # <-- new; minimal addition
    #     # temporal_stride=2)
    # )
    # x = torch.randn(12, 10, 64, 64)   # batch of 8 samples
    # out = net(x)
    # print(out.shape)   # expect: torch.Size([8, 128])
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- LOAD ON CPU, move to device explicitly ---
    simulations = torch.load("sim_data/full_comb.pt")  #, map_location="cpu")
    theta = simulations.get("theta")[:25000]  # currently on CPU
    x = simulations.get("x")[:25000]
    x = x[:, 1:, :, :]  # remove the first snapshot each

    print("x.shape:", x.shape)
    print("theta.shape:", theta.shape)

    # # move only when needed (explicit)
    theta = theta.to(device)
    x = x.to(device)
        
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

    test_data = torch.load("sim_data/test_data_full.pt")  #, map_location="cpu")
    theta_test = test_data.get("theta")
    x_test = test_data.get("x")
    x_test = x_test[:, 1:, :, :]  # remove the first snapshot each (because it is just the initial condition and doesn't contain info about the parameters)

    print("theta_test.shape:", theta_test.shape)
    print("x_test.shape:", x_test.shape)

    # # move test data to device explicitly too (if you need it on GPU)
    theta_test = theta_test.to(device)
    x_test = x_test.to(device)
    
    study_name = 'study_full_adv2'  # identifier of the study.
    storage_name = 'sqlite:///study_full_adv2.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name, directions=['minimize', 'minimize'], load_if_exists=True)
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    study.optimize(objective, callbacks=[MaxTrialsCallback(128, states=(TrialState.COMPLETE, TrialState.FAIL))],)