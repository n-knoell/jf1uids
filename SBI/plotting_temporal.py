import os
from autocvd import autocvd
autocvd(num_gpus = 1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
# load_and_sample.py
import pickle
import torch
from helper import DensityEmbedding

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from astropy import units as u
from astropy import constants as c
from jf1uids import CodeUnits
from matplotlib.lines import Line2D


plt.rcParams.update({
    'axes.labelsize': 17,   # x and y labels
    'axes.titlesize': 17,   # plot title
    # 'xtick.labelsize': 12,  # x tick labels
    # 'ytick.labelsize': 12,  # y tick labels
    'legend.fontsize': 15,  # legend text
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = os.path.join('trial20') 

# load posterior
with open(save_path + "/trained_posterior_adv_40000_trial20.pkl", "rb") as f:
    posterior = pickle.load(f)
try:
    posterior._net.to(device)   # best-effort (internal attribute, but commonly present)
except Exception:
    pass

try:
    from scipy.stats import gaussian_kde
    _HAS_KDE = True
except Exception:
    _HAS_KDE = False
from matplotlib.colors import LogNorm
from scipy.stats import norm

R_forced = 0.1
sep_in_au = 10  #upper-limit: 80, lower-limit: 5
length_temp = sep_in_au / 0.1  # before
mass_temp = 0.01 / 1  #1e-4
code_length = length_temp * u.au 
code_mass = mass_temp * u.M_sun
code_velocity = np.sqrt(c.G * code_mass / code_length).to(u.km/u.s)     # code_velocity = velocity_temp * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)


def v_code_to_real(v_code):
    return (v_code * code_units.code_velocity).to(u.km / u.s).value

def mlr_code_to_real(mlr_code):
    return (mlr_code * (code_units.code_mass / code_units.code_time)).to(u.M_sun / u.yr).value


##new limits 
mlr_lower = 8e-9 #* u.M_sun / u.yr           
mlr_upper = 1e-5 #* u.M_sun / u.yr
wind_vel_lower = 1200 #* u.km / u.s 
wind_vel_upper = 3200 #* u.km / u.s   

e_lower = 0.0
e_upper = 0.85
cos_inc_lower = 0.0
cos_inc_upper = 1.0
turbulence_strength_lower = 0.0
turbulence_strength_upper = 0.025


def corner_plot(samples, labels=None, truths=None, bins=30, figsize=None, cmap='Blues', color_option=True):  #cmap='Blues'
    """
    Minimal corner / triangle plot using matplotlib only.
    samples: (N, D) numpy array or torch tensor
    labels: list of D strings
    truths: iterable of length D (optional) for vertical/horizontal lines
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples)

    N, D = samples.shape

    if labels is None:
        labels = [f"θ{i}" for i in range(D)]

    if figsize is None:
        figsize = (1.8 * D, 1.8 * D)

    # precompute means for diagonal plotting
    means = np.mean(samples, axis=0)
    medians = np.median(samples, axis=0)
    use_log=False
    use_2log=True
    start_at_zero=False 
    min_max = True
    make_histogram = True
    n_bins = bins
    # compute per-column x-limits so every plot in a column shares the same x-range
    xlims = []
    for j in range(D):
        col = samples[:, j]
        if use_log:
            col = col[col > 0]
            if col.size == 0:
                lo, hi = 1e-3, 1.0
            else:
                lo, hi = np.percentile(col, [0.5, 99.5])
        else:
            lo, hi = np.percentile(col, [0.5, 99.5])
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        pad = 0.03 * (hi - lo)
        if start_at_zero:
            xlims.append((0, hi + pad))
        else:
            xlims.append((lo - pad, hi + pad))

    if min_max:
        xlims = [[mlr_lower, mlr_upper],[mlr_lower, mlr_upper],[wind_vel_lower, wind_vel_upper],[wind_vel_lower, wind_vel_upper], 
                [e_lower, e_upper], [cos_inc_lower, cos_inc_upper]]  #, [turbulence_strength_lower, turbulence_strength_upper]]

    fig, axes = plt.subplots(D, D, figsize=figsize, squeeze=False)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            # diagonal: marginal histogram
            if i == j:
                if make_histogram:
                    if use_log:
                        bin_edges = np.logspace(np.log10(samples[:, i].min()),
                                                np.log10(samples[:, i].max()),
                                                n_bins)
                        ax.hist(samples[:, i], bins=bin_edges, density=True, alpha=0.8)
                    elif use_2log:
                        if i < 2:
                            log_min = np.log10(samples[:, i].min())
                            log_max = np.log10(samples[:, i].max())
                            bin_edges = np.logspace(log_min, log_max, n_bins)
                            ax.hist(samples[:, i], bins=bin_edges, density=True, alpha=0.8)
                        else:
                            ax.hist(samples[:, i], bins=n_bins, density=True, alpha=0.8)
                    else:
                        ax.hist(samples[:, i], bins=n_bins, density=True, alpha=0.8)
                    # draw mean (solid) and median (dashed) lines
                    # ax.axvline(means[i], color='C1', linestyle='-', linewidth=1.2, label='Mean')
                    # ax.axvline(medians[i], color='C2', linestyle='--', linewidth=1.2, label='Median')
                    print("Mean of param", labels[i], ":", means[i])
                    # print("Median of param", labels[i], ":", medians[i])
                # replace the entire `else:` block (when make_histogram is False) with this:
                else:
                    # determine if this parameter should be treated as log based on the same flags
                    is_log_param = bool(use_log or (use_2log and i < 2))
                    data = samples[:, i]
                    mu = sd = np.nan
                    try:
                        if is_log_param:
                            data_pos = data[data > 0]
                            if data_pos.size >= 4:
                                y = np.log10(data_pos)
                                mu, sd = np.mean(y), np.std(y, ddof=0)
                                # use the precomputed column x-limits for stable plotting range
                                lo, hi = xlims[j]
                                # guard against non-positive xlim (shouldn't happen for log params)
                                lo = max(lo, 1e-12)
                                xlog = np.linspace(np.log10(lo), np.log10(hi), 400)
                                x = 10 ** xlog
                                pdf_log = norm.pdf(xlog, loc=mu, scale=sd)
                                pdf_x = pdf_log / (x * np.log(10))
                                ax.plot(x, pdf_x, color='blue', lw=1.2, label='Gaussian approx')
                                mask = (xlog >= mu - sd) & (xlog <= mu + sd)
                                ax.fill_between(x[mask], pdf_x[mask], color='blue', alpha=0.25)
                            else:
                                # not enough positive samples to fit in log-space — skip plot
                                pass
                        else:
                            # linear case
                            if data.size >= 2:
                                mu, sd = np.mean(data), np.std(data, ddof=0)
                                lo, hi = xlims[j]
                                xs = np.linspace(lo, hi, 400)
                                pdf = norm.pdf(xs, loc=mu, scale=sd)
                                ax.plot(xs, pdf, color='blue', lw=1.2, label='Gaussian approx')
                                mask = (xs >= mu - sd) & (xs <= mu + sd)
                                ax.fill_between(xs[mask], pdf[mask], color='blue', alpha=0.25)
                        print(f"mu, sd for param {labels[i]}: {mu}, {sd}")

                    except Exception as e:
                        print("exception in Gaussian approx for param", labels[i], ":", e)
                        pass

                # existing truth (dashed) if provided
                if truths is not None:
                    ax.axvline(truths[i], color='C3', linestyle='--', linewidth=1.4, label='Real value')
                ax.set_yticks([])
                if use_log:
                    ax.set_xscale('log')
                elif use_2log:
                    if i < 2:
                        ax.set_xscale('log')

                ax.set_xlim(xlims[j])
                # ax.legend()

            # lower triangle: 2D density / hexbin
            elif i > j:
                x = samples[:, j]
                y = samples[:, i]

                if color_option:
                    # use the precomputed xlims so the mesh covers the whole subplot range
                    xlo, xhi = xlims[j]
                    ylo, yhi = xlims[i]

                    def _ensure_non_degenerate(lo, hi):
                        if hi <= lo:
                            delta = max(1e-8, abs(lo) * 1e-3, 1e-6)
                            lo, hi = lo - delta, hi + delta
                        return lo, hi

                    xlo, xhi = _ensure_non_degenerate(xlo, xhi)
                    ylo, yhi = _ensure_non_degenerate(ylo, yhi)

                    # choose log-uniform edges when i < 2 and ranges are positive; otherwise linear edges
                    if i < 2 and (xlo > 0 and xhi > 0 and ylo > 0 and yhi > 0):
                        # log-uniform edges
                        xedges = np.logspace(np.log10(xlo), np.log10(xhi), bins + 1)
                        yedges = np.logspace(np.log10(ylo), np.log10(yhi), bins + 1)
                        print("XXXXXXXXXXXX")
                    else:
                        # linear edges (fallback or normal case)
                        xedges = np.linspace(xlo, xhi, bins + 1)
                        yedges = np.linspace(ylo, yhi, bins + 1)

                    # compute histogram using those edges (note bins=[xedges, yedges])
                    H, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], density=True)

                    # replace zero bins with the smallest positive value so colormap is defined everywhere
                    if np.any(H > 0):
                        min_pos = H[H > 0].min()
                        H_plot = H.copy()
                        H_plot[H_plot == 0] = min_pos
                    else:
                        # no positive bins (very unlikely) — use a tiny epsilon everywhere
                        eps = 1e-12
                        H_plot = H + eps
                        min_pos = H_plot.min()

                    # plot the histogram (transpose H so orientation matches x/y)
                    pcm = ax.pcolormesh(xedges, yedges, H_plot.T, shading='auto') #, cmap=cmap)

                    # make sure axes exactly follow the xlims used to create edges
                    ax.set_xlim(xlo, xhi)
                    ax.set_ylim(ylo, yhi)
                else:
                    try:
                        ax.hist2d(x, y, bins=bins, density=True, cmap=cmap)
                    except Exception:
                        print("hist2d failed, using scatter")
                        ax.plot(x, y, '.', ms=2, alpha=0.4)
            
                
                # smooth contour if KDE available
                if _HAS_KDE:
                    xy = np.vstack([x, y])
                    try:
                        kde = gaussian_kde(xy)
                        xmin, xmax = np.percentile(x, [0.5, 99.5])
                        ymin, ymax = np.percentile(y, [0.5, 99.5])
                        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                        ax.contour(X, Y, Z, colors='k', linewidths=0.7, levels=4, alpha=0.7)
                        
                    except Exception:
                        pass
                if truths is not None:
                    ax.plot(truths[j], truths[i], 'r*', ms=8)
                if use_log:
                    ax.set_xscale('log')
                elif use_2log:
                    if j < 2:
                        ax.set_xscale('log')
                ax.set_xlim(xlims[j])

                # compute and print Pearson r (minimal dependency: use numpy)
                try:
                    r, p = pearsonr(x, y)
                    ax.text(0.05, 0.95,
                            f"r = {r:.2f}",  #\np = {p:.1e}
                            transform=ax.transAxes,
                            fontsize=12,
                            va='top',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                    )
                except Exception:
                    pass
                
            # upper triangle: blank
            else:
                ax.axis('off')

            # axis labels only on leftmost/bottom panels
            if i < D - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(labels[j])
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(labels[i])
    legend_elements = [Line2D([0], [0], color='r', linestyle='-', lw=2, label='true value')]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=15)
    plt.tight_layout()
    return fig, axes



def image_grid(images, ncols=6, figsize=None, cmap='viridis', vmin=None, vmax=None, titles=None):
    """
    images: list or array of (H, W) 2D arrays. If images is shape (N, H, W) ok.
    Shows images in a grid.
    """
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[None, ...]
    N = images.shape[0]
    ncols = min(ncols, N)
    nrows = int(np.ceil(N / ncols))
    if figsize is None:
        figsize = (2.2 * ncols, 2.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        ax = axes[i]
        ax.axis('off')
        if i < N:
            norm = LogNorm(vmin=vmin, vmax=vmax)
            im = ax.imshow(images[i], origin='lower', cmap=cmap, norm=norm)
            if titles is not None:
                ax.set_title(titles[i])
    # add colorbar on the right
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cax)
    return fig, axes


def posterior_predictive_grid(posterior_samples, simulator_fn, nplot=9, ncols=3, random_seed=None):
    """
    Posterior predictive: run the simulator (fast) on several posterior samples and show density slices.
    posterior_samples: (N, D) numpy array or torch
    simulator_fn: function(theta) -> 2D array (H, W) or an image
    """
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(posterior_samples), size=min(nplot, len(posterior_samples)), replace=False)
    selected = posterior_samples[idx]
    imgs = []
    for th in selected:
        imgs.append(np.asarray(simulator_fn(th)))
    fig, axes = image_grid(imgs, ncols=ncols)
    return fig, axes


def exp_theta(theta):
    """
    Convert theta samples from inference space to physical space.
    theta: (N, 4) array-like or torch tensor with parameters in inference space
    """
    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy()

    theta = np.asarray(theta)
    phys = np.empty_like(theta)

    for i in range(theta.shape[1]):
        phys[:, i] = np.exp(theta[:, i])
    
    return phys

def exp2_theta(theta):
    """
    Convert theta samples from inference space to physical space.
    theta: (N, 4) array-like or torch tensor with parameters in inference space
    """
    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy()

    theta = np.asarray(theta)
    phys = np.empty_like(theta)

    for i in range(theta.shape[1]):
        if i <= 1:
            phys[:, i] = np.exp(theta[:, i])
        else:
            phys[:, i] = theta[:, i]
    
    return phys


# ---- Example usage with your saved files ----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = torch.load("sim_data/test_data_960_full.pt", map_location="cpu")

    theta_test = test_data.get("theta")
    x_test = test_data.get("x") 
    x_test = x_test[:,1:,:,:]
    theta_test = theta_test.to(device)
    x_test = x_test.to(device)
    test_num = 2
    x_o = x_test[test_num]  #
    theta_o = theta_test[test_num]
    print("THETA of ref sim:", theta_o)

    ##IF u want to LOAD SPECIFIC EXAMPLE SIMULATIONS
    specific_example = True
    if specific_example:
        name = "diff"
        data = np.load(f"ref_obs_{name}.npz", allow_pickle=True)
        one_run = torch.as_tensor(data["one_run"], device=device)
        theta_jax = torch.as_tensor(data["theta"], device=device)

        theta_o = theta_jax
        x_o = one_run[1:,:,:]
        print("Loaded specific example with theta:", theta_o)

    num_posterior_samples = 50000
    posterior_samples = posterior.sample((num_posterior_samples,), x=x_o)  # shape (num_posterior_samples, D)
    posterior_samples = exp2_theta(posterior_samples)

    print("post sample: ", posterior_samples[0])

    # posterior_samples = posterior_samples.detach().cpu().numpy()
    # Convert posterior samples to real values
    posterior_samples_real = np.array([
        [mlr_code_to_real(posterior_samples[i, 0]), mlr_code_to_real(posterior_samples[i, 1]),
        v_code_to_real(posterior_samples[i, 2]), v_code_to_real(posterior_samples[i, 3]),
        posterior_samples[i, 4],posterior_samples[i, 5]]   #,posterior_samples[i, 6]]
        for i in range(len(posterior_samples))
    ])

    theta_o = theta_o.detach().cpu().numpy()
    theta_o_real = np.array([
        mlr_code_to_real(np.exp(theta_o[0])), mlr_code_to_real(np.exp(theta_o[1])),
        v_code_to_real(theta_o[2]), v_code_to_real(theta_o[3]), theta_o[4], theta_o[5], theta_o[6]]
    )

    #plot everything except turb param:
    # posterior_samples_real = posterior_samples_real[:6]
    print("Reference theta (real):", theta_o_real)
    theta_o_real = theta_o_real[:6]
    fig, axes = corner_plot(posterior_samples_real, bins=80, figsize=(15,15),
                            labels=[r"$\log \dot M_1$",r"$\log \dot M_2$",r"$v_{\inf,1}$",r"$v_{\inf,2}$", r"$e$", r"$\cos (i)$"],truths=theta_o_real, color_option=False)
    # fig.suptitle("Posterior samples", fontsize=14)
    if specific_example:
        plt.savefig(save_path + f"/posterior_{name}_full.png", dpi=300)
    else:
        plt.savefig(save_path + "/posterior_full_"+str(test_num)+".png", dpi=300)