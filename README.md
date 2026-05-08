# Neural Posterior Estimation of Colliding-Wind Binary Parameters from Hα Time Series

> ⚠️ **Anonymous repository for double-blind peer review.**

---

## Overview

This repository contains the full pipeline used to perform amortized **simulation-based inference (SBI)** of colliding-wind binary (CWB) parameters from short Hα photon-count time series. Given a 10-frame, 64×64 noisy Hα image cube of a CWB, the trained model returns an approximate posterior over seven physical parameters:
 `Ṁ₁`, `Ṁ₂` (mass-loss rates), `v∞,₁`, `v∞,₂` (Terminal wind velocities), `e` (Orbital eccentricity), `i` (Inclination), `η` (Turbulence-to-wind luminosity ratio)

The pipeline consists of four stages: (i) a 3D differentiable hydrodynamics + N-body forward simulator, (ii) an Hα emissivity and detector-noise pipeline that produces synthetic photon-count maps, (iii) a factorised spatio-temporal CNN embedding, and (iv) a neural spline flow trained with NPE-C (Greenberg et al., 2019) using the [`sbi`](https://github.com/sbi-dev/sbi) toolkit (Boelts et al., 2025).

---

## Forward simulator: modified `astronomix`

The hydrodynamics are solved with a **modified version of `astronomix`**, a JAX-based, auto-differentiable, conservative finite-volume fluid solver.

**References:**
- *astronomix — differentiable MHD in JAX*. Zenodo (2025). DOI: [10.5281/zenodo.17782162](https://zenodo.org/doi/10.5281/zenodo.17782162)

The upstream solver provides the differentiable finite-volume hydrodynamics core. The complete list of modifications to `astronomix` is documented below.

### Modifications

1. **Two-step gravity coupling for a moving binary.**
   - Stars: integrated with an explicit fourth-order Runge–Kutta N-body integrator. Gas gravity onto stars neglected relative to the stellar masses.
   - Gas: feels the stars through a Poisson solve in which the stellar masses are deposited onto the grid using a nearest-grid-point (NGP) kernel each step.
   This couples the upstream hydro solver to a moving binary while keeping the gravitational sourcing consistent on the Eulerian grid.

2. **Empirical mass ↔ mass-loss-rate coupling.** The sampled mass-loss rate `Ṁ` is mapped to a stellar mass `M ~ N(M₀(Ṁ), 0.05 M₀(Ṁ))` via an interpolated empirical relation derived from the rotating stellar-evolution grids of Ekström et al. (2012), with wind-parameter post-processing following Haid et al. (2018).

3. **Hα emissivity post-processor.** A new module computes the Hα volume emissivity from the simulated `(ρ, P)` fields, assuming a fully photoionised H II region (cf. Green et al., 2019), via interpolation in a table from Osterbrock (1989):
   `j_Hα = 2.63·10⁻³³ · n_e n_H / T^0.9   [erg cm⁻³ s⁻¹ arcsec⁻²]`

4. **Line-of-sight projection (optically thin limit).** The 3D emissivity cube is integrated along the line of sight, to produce a 2D intensity map per snapshot.

5. **Detector / observation model.** Intensity is converted to expected photon counts using representative instrument parameters, followed by a per-pixel noise model. Noise is resampled per snapshot, per epoch during training.

---

## License & data

Code is released for review purposes only under this anonymous repository.

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
- Talts, S. et al. *Validating Bayesian inference algorithms with simulation-based calibration.* [arXiv:1804.06788](https://arxiv.org/abs/1804.06788) (2018).
