# Neural Posterior Estimation of Colliding-Wind Binary Parameters from Hα Time Series

> ⚠️ **Anonymous repository for double-blind peer review.**
> This repository hosts the code accompanying the paper *"Neural Posterior Estimation of Colliding-Wind Binary Parameters from Hα Time Series"*, submitted to the **AI4Physics Workshop at ICML 2026**. All identifying information (authors, affiliations, links to non-anonymous code or data) has been removed for the duration of the review process. A non-anonymous, citable release will replace this repository upon acceptance.

---

## Overview

This repository contains the full pipeline used to perform amortized **simulation-based inference (SBI)** of colliding-wind binary (CWB) parameters from short Hα photon-count time series. Given a 10-frame, 64×64 noisy Hα image cube of a CWB, the trained model returns an approximate posterior over seven physical parameters:

| Symbol | Meaning | Prior |
|---|---|---|
| `Ṁ₁`, `Ṁ₂` | Individual mass-loss rates [M☉/yr] | log-uniform, 8·10⁻⁹ – 10⁻⁵ |
| `v∞,₁`, `v∞,₂` | Terminal wind velocities [km/s] | uniform, 1200 – 3200 |
| `e` | Orbital eccentricity | uniform, 0 – 0.85 |
| `i` | Inclination [deg] | isotropic, 0 – 90 |
| `η` | Turbulence-to-wind luminosity ratio | uniform, 0 – 0.025 |

The pipeline consists of four stages: (i) a 3D differentiable hydrodynamics + N-body forward simulator, (ii) an Hα emissivity and detector-noise pipeline that produces synthetic photon-count maps, (iii) a factorised spatio-temporal CNN embedding, and (iv) a neural spline flow trained with **NPE-C** (Greenberg et al., 2019) using the [`sbi`](https://github.com/sbi-dev/sbi) toolkit (Boelts et al., 2025).

---

## Forward simulator: modified `astronomix`

The hydrodynamics are solved with a **modified version of `astronomix`** (formerly `jf1uids`), a JAX-based, auto-differentiable, conservative finite-volume fluid solver.

**Upstream references:**
- Storcks, L. *astronomix — differentiable MHD in JAX*. Zenodo (2025). DOI: [10.5281/zenodo.17782162](https://zenodo.org/doi/10.5281/zenodo.17782162)
- Storcks, L. & Buck, T. *Differentiable conservative radially symmetric fluid simulations and stellar winds — jf1uids* (2024). [arXiv:2410.23093](https://arxiv.org/abs/2410.23093)

The upstream solver provides the differentiable finite-volume hydrodynamics core. On top of it, this work adds a CWB-specific physics layer and an observation-generation layer. The complete list of modifications relative to upstream `astronomix` is documented below.

### Modifications introduced in this work

1. **Energy-and-mass wind injection sources.** Each star is represented as a point source with a spherical injection region of volume `V`. Mass is deposited at rate `Ṁₛ / V` and kinetic energy at rate `½ v²∞,ₛ Ṁₛ / V`. The momentum source then arises self-consistently through the induced pressure gradient, so no separate momentum injection term is required.

2. **Two-step gravity coupling for a moving binary.**
   - **Stars:** integrated with an explicit fourth-order Runge–Kutta N-body integrator. Gas self-gravity is neglected relative to the stellar masses.
   - **Gas:** feels the stars through a Poisson solve in which the stellar masses are deposited onto the grid using a nearest-grid-point kernel each step.
   This couples the upstream hydro solver to a moving binary while keeping the gravitational sourcing consistent on the Eulerian grid.

3. **Forced random turbulence parameterised by `η`.** A driven turbulence scheme adopted and modified from Seo & Ryu (2023, [DOI](https://doi.org/10.3847/1538-4357/acdf4b)) is added, with amplitude controlled by the dimensionless parameter `η = Ė_turb / ⟨L_wind⟩` — the fraction of additional forced turbulence relative to the mean injected wind luminosity.

4. **Empirical mass ↔ mass-loss-rate coupling.** The sampled mass-loss rate `Ṁ` is mapped to a stellar mass `M ~ N(M₀(Ṁ), 0.05 M₀(Ṁ))` via an interpolated empirical relation derived from the rotating stellar-evolution grids of Ekström et al. (2012), with wind-parameter post-processing following Haid et al. (2018).

5. **Hα emissivity post-processor.** A new module computes the Hα volume emissivity from the simulated `(ρ, P)` fields, assuming a fully photoionised H II region (cf. Green et al., 2019), via interpolation in a table from Osterbrock (1989):
   `j_Hα = 2.63·10⁻³³ · n_e n_H / T^0.9   [erg cm⁻³ s⁻¹ arcsec⁻²]`
   with `n_e = 0.86 ρ/m_p`, `n_H = 0.71 ρ/m_p`, and the temperature obtained from the ideal gas law.

6. **Line-of-sight projection (optically thin limit).** The 3D emissivity cube is integrated along the line of sight, `J = ∫ j_Hα dl`, to produce a 2D intensity map per snapshot.

7. **Detector / observation model.** Intensity is converted to expected photon counts using representative instrument parameters (`D = 2.4 m`, `A_ap = 0.04 arcsec²`, `t_exp = 600 s`, `η_tel = 0.11`), followed by a per-pixel noise model combining Poisson photon shot noise (`σ = √N`) and a flat-field calibration term with fractional amplitude `ε_flat = 0.01`. Noise is **resampled per snapshot, per epoch** during training, so each underlying simulation is seen with many independent noise realisations.

8. **Snapshot / time-series output.** Each run is configured to advance for `T_end = 5 yr` on an `N = 64³` Cartesian grid and emit **10 equally spaced snapshots** matched to the encoder's expected input shape `(T=10, H=64, W=64)`.

9. **CWB-specific initial conditions.** Uniform ambient density `ρ₀ = 2 m_p cm⁻³` and temperature `T₀ = 15 000 K`, representative of the warm ionised ISM around massive binaries.

---

## Repository layout

```
.
├── simulator/        # Modified astronomix + N-body integrator + wind/turbulence sources
├── observation/      # Hα emissivity, line-of-sight projection, detector & noise model
├── inference/        # Spatio-temporal CNN embedding + neural spline flow (NPE-C via sbi)
├── hpo/              # Optuna NSGA-II hyperparameter search (validation NLL ⊕ TARP deviation)
├── calibration/      # TARP and SBC diagnostics
├── baselines/        # ABC rejection sampler with the 62-D hand-crafted summary
├── scripts/          # Training, inference, and figure-reproduction entry points
└── configs/          # Priors, simulator settings, network/flow hyperparameters
```

## Pipeline at a glance

1. **Sample** `θ ~ p(θ)` from the priors in the table above.
2. **Simulate** with modified `astronomix` for 5 yr on a 64³ grid, dumping 10 snapshots.
3. **Render** Hα emissivity, project along the line of sight, apply detector + noise model → `x ∈ ℝ^{10×64×64}`.
4. **Embed** with a shared 2D CNN per frame → temporal 1D CNN → FC head.
5. **Train** a neural spline flow `q_φ(θ | f_ψ(x))` jointly with the embedding, minimising the NPE-C loss on `~40 000` `(θᵢ, xᵢ)` pairs.
6. **Evaluate** posterior calibration with TARP (joint) and SBC (marginal).

## Reproducing the paper

The selected hyperparameters from the 128-trial Optuna NSGA-II Pareto search (jointly minimising validation NLL and mean TARP deviation) are:

| Hyperparameter | Value |
|---|---|
| Conv blocks `L` | 3 |
| Base channels `C₁` | 32 |
| Final FC width `d_fc` | 128 |
| Temporal layers `L_t` | 2 |
| Temporal first-layer stride `s_t` | 1 |
| Temporal pool bins `T_out` | 2 |
| Flow transforms `N_tr` | 20 |
| Flow hidden features `H_flow` | 7 |

Reference run: `40 000` training simulations, `1 000` held-out test simulations, single NVIDIA H200 GPU (≈ 4 100 simulations/day).

## Requirements

- Python ≥ 3.10
- JAX (with a CUDA-enabled build for GPU runs)
- A modified copy of `astronomix` (vendored / pinned in `simulator/`)
- [`sbi`](https://github.com/sbi-dev/sbi) (Boelts et al., 2025), PyTorch, Optuna, NumPy, SciPy, Matplotlib

## License & data

Code is released for **review purposes only** under this anonymous repository. A permissive open-source license, full Zenodo data release of the `40 000` simulation set, and links to the de-anonymised repository will accompany publication.

## Key references

- Boelts, J. et al. *sbi reloaded: a toolkit for simulation-based inference workflows.* JOSS, 10(108):7754 (2025).
- Cantó, J., Raga, A. C. & Wilkin, F. P. *Exact, Algebraic Solutions of the Thin-Shell Two-Wind Interaction Problem.* ApJ 469:729 (1996).
- Durkan, C. et al. *Neural Spline Flows.* [arXiv:1906.04032](https://arxiv.org/abs/1906.04032) (2019).
- Ekström, S. et al. *Grids of stellar models with rotation.* A&A 537:A146 (2012).
- Greenberg, D., Nonnenmacher, M. & Macke, J. *Automatic Posterior Transformation for Likelihood-Free Inference.* ICML (2019).
- Haid, S. et al. *The relative impact of photoionizing radiation and stellar winds on different environments.* MNRAS 478(4):4799 (2018).
- Lemos, P. et al. *Sampling-based accuracy testing of posterior estimators (TARP).* [arXiv:2302.03026](https://arxiv.org/abs/2302.03026) (2023).
- Osterbrock, D. E. *Astrophysics of Gaseous Nebulae and Active Galactic Nuclei* (1989).
- Seo, J. & Ryu, D. *HOW-MHD: A High-Order WENO-Based MHD Code…* ApJ 953(1):39 (2023).
- Storcks, L. *astronomix — differentiable MHD in JAX.* Zenodo (2025), [10.5281/zenodo.17782162](https://zenodo.org/doi/10.5281/zenodo.17782162).
- Storcks, L. & Buck, T. *jf1uids — differentiable conservative radially symmetric fluid simulations and stellar winds.* [arXiv:2410.23093](https://arxiv.org/abs/2410.23093) (2024).
- Talts, S. et al. *Validating Bayesian inference algorithms with simulation-based calibration.* [arXiv:1804.06788](https://arxiv.org/abs/1804.06788) (2018).
