import os
from autocvd import autocvd
autocvd(num_gpus = 1, interval = 1)

# numerics
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax import random

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from jf1uids import CodeUnits
from astropy import units as u
import astropy.constants as const
from astropy.constants import m_p
import torch 

from matplotlib.colors import LogNorm
import time

R_forced = 0.1
sep_in_au = 10  #upper-limit: 80, lower-limit: 5
length_temp = sep_in_au / 0.1  # before
mass_temp = 0.01 / 1  #1e-4
code_length = length_temp * u.au 
code_mass = mass_temp * u.M_sun
code_velocity = np.sqrt(const.G * code_mass / code_length).to(u.km/u.s)     # code_velocity = velocity_temp * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

code_energy = code_mass * code_length**2 / code_units.code_time**2
code_emissivity_unit = code_energy / code_units.code_time / code_length**3 / u.arcsec
phys_unit = u.erg / u.s / u.cm**3 / u.arcsec
conv = (1 * phys_unit).to(code_emissivity_unit).value
h = const.h.to(code_units.code_mass*code_units.code_length**2/code_units.code_time).value
c = const.c.to(code_units.code_velocity).value
kB = const.k_B.to(code_units.code_pressure * code_units.code_length**3 / u.K).value
m_p = const.m_p.to(code_units.code_mass).value
M_P = const.m_p.value
mu_val = 0.61
X = 0.71            # hydrogen mass fraction
Y = 0.27            # helium mass fraction
invR_specific = (mu_val * m_p) / kB 
h_div_kb = h / kB

def v_code_to_real(v_code):
    return (v_code * code_units.code_velocity).to(u.km / u.s).value

def mlr_code_to_real(mlr_code):
    return (mlr_code * (code_units.code_mass / code_units.code_time)).to(u.M_sun / u.yr).value

C_phys = 2.63e-33            # physical prefactor in original jHalpha formula
code_len_cm = (1.0 * code_units.code_length).to(u.cm).value  # one code-length in cm

# (v_code * code_units.code_velocity).to(u.km / u.s).value
code_vol_cm3 = code_len_cm**3

J_PREFAC = conv * C_phys / (code_vol_cm3**2)
pref = (J_PREFAC / (m_p ** 2))
elec_per_H = 1.0 + (Y / (2.0 * X))

@jit
def build_j_map(rho, P):
    """
    Compute H-alpha emissivity map staying entirely in code units.
    Inputs:
      - rho: array-like (code density)
      - P: array-like (code pressure)

    Returns:
      - j_map_code: emissivity map in code emissivity units
    """

    denom = rho / invR_specific
    T_code = P / denom
    rhoX = rho * X

    j_map = pref * (rhoX ** 2) * elec_per_H / (T_code ** 0.9)

    return j_map


#### rotation and projection code (mostly from earlier pipeline.py, but with some edits) --- IGNORE ---
# Minimal grid parameters from 1D coords
@jit
def grid_params_from_1d(x1, y1, z1):
    nx = x1.shape[0]; ny = y1.shape[0]; nz = z1.shape[0]
    x0, y0, z0 = x1[0], y1[0], z1[0]
    dx = x1[1] - x1[0] if nx > 1 else 1.0
    dy = y1[1] - y1[0] if ny > 1 else 1.0
    dz = z1[1] - z1[0] if nz > 1 else 1.0
    bounds_min = jnp.array([x1[0], y1[0], z1[0]])
    bounds_max = jnp.array([x1[-1], y1[-1], z1[-1]])
    center = (bounds_min + bounds_max) / 2.0
    return dict(nx=nx, ny=ny, nz=nz, x0=x0, y0=y0, z0=z0,
                dx=dx, dy=dy, dz=dz, bounds_min=bounds_min,
                bounds_max=bounds_max, center=center)

# Trilinear sampler: pts shape (M,3) -> (M,)
@jit
def trilinear_sample(emiss, params, pts):
    x0, y0, z0 = params['x0'], params['y0'], params['z0']
    dx, dy, dz = params['dx'], params['dy'], params['dz']
    nx, ny, nz = params['nx'], params['ny'], params['nz']

    fx = (pts[:, 0] - x0) / dx
    fy = (pts[:, 1] - y0) / dy
    fz = (pts[:, 2] - z0) / dz

    inside = (fx >= 0) & (fx <= (nx - 1)) & (fy >= 0) & (fy <= (ny - 1)) & (fz >= 0) & (fz <= (nz - 1))

    ix = jnp.floor(jnp.clip(fx, 0, nx - 2)).astype(jnp.int32)
    iy = jnp.floor(jnp.clip(fy, 0, ny - 2)).astype(jnp.int32)
    iz = jnp.floor(jnp.clip(fz, 0, nz - 2)).astype(jnp.int32)

    wx = (fx - ix.astype(jnp.float32)).reshape(-1, 1)
    wy = (fy - iy.astype(jnp.float32)).reshape(-1, 1)
    wz = (fz - iz.astype(jnp.float32)).reshape(-1, 1)

    # 8 corners
    c000 = emiss[ix,   iy,   iz  ]
    c100 = emiss[ix+1, iy,   iz  ]
    c010 = emiss[ix,   iy+1, iz  ]
    c001 = emiss[ix,   iy,   iz+1]
    c101 = emiss[ix+1, iy,   iz+1]
    c011 = emiss[ix,   iy+1, iz+1]
    c110 = emiss[ix+1, iy+1, iz  ]
    c111 = emiss[ix+1, iy+1, iz+1]

    def _col(a): return a.reshape(-1,1)
    c000,c100,c010,c001 = _col(c000),_col(c100),_col(c010),_col(c001)
    c101,c011,c110,c111 = _col(c101),_col(c011),_col(c110),_col(c111)

    c00 = c000*(1-wx) + c100*wx
    c01 = c001*(1-wx) + c101*wx
    c10 = c010*(1-wx) + c110*wx
    c11 = c011*(1-wx) + c111*wx

    c0 = c00*(1-wy) + c10*wy
    c1 = c01*(1-wy) + c11*wy

    out = (c0*(1-wz) + c1*wz).reshape(-1)
    return out * inside.astype(out.dtype)

# Integrator for one ray (jit)
def make_ray_integrator(emiss, params, n_steps):
    # precompute 8 corners of box
    mn, mx = params['bounds_min'], params['bounds_max']
    corners = jnp.stack(jnp.meshgrid(jnp.array([mn[0], mx[0]]),
                                     jnp.array([mn[1], mx[1]]),
                                     jnp.array([mn[2], mx[2]]),
                                     indexing='ij'), axis=-1).reshape(-1,3)  # (8,3)

    @jit
    def integrate(origin, nvec):
        tvals = jnp.dot(corners - origin, nvec)   # projection of corners onto ray
        tmin = jnp.min(tvals); tmax = jnp.max(tvals)
        # zero if no span
        dt_total = tmax - tmin
        def integrate_path():
            ts = jnp.linspace(tmin, tmax, n_steps)
            pts = origin[None,:] + ts[:,None]*nvec[None,:]   # (n_steps,3)
            vals = trilinear_sample(emiss, params, pts)     # (n_steps,)
            dt = dt_total / (n_steps - 1)
            return jnp.sum(0.5*(vals[:-1] + vals[1:])) * dt
        return jax.lax.cond(dt_total > 0.0, integrate_path, lambda: 0.0)
    return integrate

def project_at_theta_zaxis(emiss, x1, y1, z1, theta, nu=None, nv=None, n_steps=None):
    """
    Raytrace with LOS rotated around the z-axis.
    emiss: (nx,ny,nz)
    x1,y1,z1: 1D coords
    theta: angle (rad) between LOS and +y axis (0 -> +y, pi/2 -> +x)
    returns img (nv,nu), u_coords (nu,), v_coords (nv,)
    """
    params = grid_params_from_1d(x1, y1, z1)
    nx, ny, nz = params['nx'], params['ny'], params['nz']
    if nu is None: nu = nx
    if nv is None: nv = nz   # default: vertical image axis = z
    if n_steps is None: n_steps = max(nx, ny, nz)

    # LOS in x-y plane (rotation about z)
    nvec = jnp.array([jnp.sin(theta), jnp.cos(theta), 0.0])

    # choose uvec perpendicular to nvec (in x-y plane), vvec = n x u -> +z
    uvec = jnp.array([-jnp.cos(theta), jnp.sin(theta), 0.0])
    # normalize uvec (optional but safe)
    uvec = uvec / (jnp.linalg.norm(uvec) + 1e-16)
    vvec = jnp.cross(nvec, uvec)
    vvec = vvec / (jnp.linalg.norm(vvec) + 1e-16)

    # compute u/v extents relative to box center (important)
    mn, mx = params['bounds_min'], params['bounds_max']
    corners = jnp.stack(jnp.meshgrid(jnp.array([mn[0], mx[0]]),
                                     jnp.array([mn[1], mx[1]]),
                                     jnp.array([mn[2], mx[2]]),
                                     indexing='ij'), axis=-1).reshape(-1,3)
    center = params['center']
    uvals = jnp.dot(corners - center, uvec)
    vvals = jnp.dot(corners - center, vvec)
    u_min, u_max = jnp.min(uvals), jnp.max(uvals)
    v_min, v_max = jnp.min(vvals), jnp.max(vvals)

    u_coords = jnp.linspace(u_min, u_max, nu)
    v_coords = jnp.linspace(v_min, v_max, nv)

    U, V = jnp.meshgrid(u_coords, v_coords, indexing='xy')   # (nv,nu)
    origins = center[None,:] + U.reshape(-1,1)*uvec[None,:] + V.reshape(-1,1)*vvec[None,:]

    integrator = make_ray_integrator(emiss, params, n_steps)
    vals = vmap(lambda o: integrator(o, nvec))(origins)
    img = vals.reshape(nv, nu)
    return img, u_coords, v_coords

# FAST SHORTCUT: if theta == pi/2 and you want image on solver grid:
@jit
def project_theta_pi2_fast(emiss, dz):
    # uniform dz assumed; returns (nx,ny)
    return jnp.sum(emiss, axis=2) * dz

_h_inv = 1.0 / h
sqrt_h_inv = (_h_inv)**0.5 
ap = 0.04                                           # diameter aperture area (arcsec^2)
D = (2.4*u.m).to(code_units.code_length).value           # 2.4m telescope
t = (600.0 * u.s).to(code_units.code_time).value          # 600 s
eta = 0.11                                          # 11% throughput
wavelength_code = (656.28 * u.nm).to(code_units.code_length).value

area_tel_code = jnp.pi * (D ** 2) / 4.0    # 8.085772509518466e-26
E_div_h = c / wavelength_code  # # 8.542145847859683e+22
temp2 = ap * area_tel_code * sqrt_h_inv #+e25
temp3 = E_div_h * h**0.5   #e-19

@jax.jit
def sb_to_photons_code(surface_sb_code):
    """
    Convert surface brightness (in *code* surface-brightness units)
    to detected photon counts — entirely in code units.

    All inputs must already be expressed in *code* units:
      - surface_sb_code : code_energy / code_time / code_length^2 / arcsec^2

    Returns:
      detected_photons : photons (dimensionless number) — same shape as surface_sb_code
    """
    energy_rate_code = surface_sb_code * temp2    #e-4
    photons_per_time = energy_rate_code / temp3  #e15
    detected_photons = photons_per_time * eta * t

    return detected_photons

def add_detector_noise_jax(image_photons,
                           sky_photons_per_pixel=0.0,
                           exposure_time_s=600.0,
                           qe=1.0,
                           dark_current_e_per_pix_per_s=0.001,
                           read_noise_e_rms=5.0,
                           flat_field_rms=0.01,
                           gain_e_per_adu=1.0,
                           key=None,
                           return_adu=False):
    """
    noise model. Inputs:
      - image_photons: expected source photons per pixel for full exposure (array or scalar)
      - sky_photons_per_pixel: expected sky photons per pixel in same exposure
      - qe: quantum efficiency (photons -> electrons). If QE already in `eta`, set qe=1.0.
      - dark_current_e_per_pix_per_s: electrons/pix/s
      - read_noise_e_rms: electrons RMS (Gaussian)
      - flat_field_rms: fractional RMS for multiplicative flat-field (applied before Poisson)
      - key: jax.random.PRNGKey (required)
      - return_adu: if True return ADU (float), otherwise electrons

    Returns:
      noisy_image, new_key
      - noisy_image: array (electrons or ADU), same shape as image_photons
      - new_key: jax.random.PRNGKey for further draws
    """
    img = image_photons
    sky = sky_photons_per_pixel

    # advance RNG and split into subkeys
    k0, k1, k2, k3 = random.split(key, 4)

    # 1) flat-field multiplicative variation (mean=1, sigma=flat_field_rms)
    if flat_field_rms > 0.0:
        flat = random.normal(k1, shape=img.shape) * jnp.asarray(flat_field_rms) + 1.0
        flat = jnp.clip(flat, 0.5, 1.5)
    else:
        flat = 1.0

    # 2) expected photons after flat-field (source + sky) * flat
    expected_photons = (img + sky) * flat
    expected_photons = jnp.clip(expected_photons, a_min=0.0)  # safety

    # 3) convert photons -> expected electrons (QE) and add dark current expectation
    expected_electrons = expected_photons * jnp.asarray(qe)
    dark_expected = jnp.asarray(dark_current_e_per_pix_per_s) * jnp.asarray(exposure_time_s)
    expected_electrons = expected_electrons + dark_expected
    expected_electrons = jnp.clip(expected_electrons, a_min=0.0)

    # 4) Poisson sampling (photons + dark shot noise)
    noisy_electrons_poisson = random.poisson(k2, lam=expected_electrons)

    # 5) add Gaussian read noise (zero-mean)
    if read_noise_e_rms > 0.0:
        read_noise = random.normal(k3, shape=noisy_electrons_poisson.shape) * jnp.asarray(read_noise_e_rms)
    else:
        read_noise = 0.0

    noisy_electrons = noisy_electrons_poisson + read_noise
    noisy_electrons = jnp.clip(noisy_electrons, a_min=0.0)

    if return_adu:
        noisy_out = noisy_electrons / jnp.asarray(gain_e_per_adu)
    else:
        noisy_out = noisy_electrons

    return noisy_out, k0