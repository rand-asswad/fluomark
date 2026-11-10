from os.path import dirname, realpath
import sys
dir_path = dirname(realpath(__file__))
sys.path += [dir_path, dirname(dir_path)]

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from fluomark.kinetics import monod, monod_inverse, monod_derivative
from fluomark.mpl_utils import set_mpl_global, legend

# PARAMETERS ===================================================================

# model parameters
default_parameters = {
    "s_in": 2.0,
    "beta": 0.8,
    "gamma": 0.45,
    "mu_max": 0.7,
    "ks": 0.002,
    "noise_std": 0.01,
}

# initial state
default_x0 = np.array([0.0, 1.0, 0.0])

# default inputs
def dilution_func(t, t_switch=2.0, d0=1.5, d_star=0.15):
    if np.isscalar(t):
        return d0 if t < t_switch else d_star
    d = d_star * np.ones_like(t)
    d[t < t_switch] = d0
    return d

def allocation_func(t, t_switch=24, alpha_max=0.3, r=0.25):
    return alpha_max / (1.0 + np.exp(-r * (t - t_switch)))

# EQUILIBRIUM ==================================================================

def washout_dilution(alpha, params=default_parameters):
    return (1.0 - alpha) * monod(params['s_in'], params['mu_max'], params['ks'])

def s_eq(alpha, d, params=default_parameters):
    mu_inv = lambda r: monod_inverse(r, params['mu_max'], params['ks'])
    if np.isscalar(alpha):
        if np.abs(alpha - 1.0) < 1e-10:
            return np.nan
        return mu_inv(d / (1.0 - alpha))
    r = np.divide(d, 1.0 - alpha, out=np.full_like(alpha, np.nan),
                  where=np.abs(1.0 - alpha) > 1e-10)
    return mu_inv(r)

def c_star(alpha, d, params=default_parameters):
    s = s_eq(alpha, d, params)
    return alpha * monod(s, params['mu_max'], params['ks']) - \
        (1.0-alpha) * monod_derivative(s, params['mu_max'], params['ks']) * (params['s_in'] - s)

# SIMULATION ===================================================================

# PLOTS ========================================================================

def operational_diagram(params=default_parameters, point=None):
    alpha = np.linspace(0, 1, 500)
    d_max = 1.02 * monod(params['s_in'], params['mu_max'], params['ks'])
    d = np.linspace(0, d_max, 500)
    A, D = np.meshgrid(alpha, d)

    D_washout = washout_dilution(A, params)
    valid = D < D_washout

    C = np.full_like(A, np.nan, dtype=float)
    C[valid] = c_star(A[valid], D[valid], params)

    fig, ax = plt.subplots(figsize=(5, 2.6))

    # contraction function contours
    contourf = ax.contourf(A, D, C, levels=30)
    cbar = fig.colorbar(contourf, ax=ax)
    cbar.set_label(r"$c^\star(\alpha,d)$")
    cbar.ax.locator_params(nbins=5)

    # washout
    alpha_line = np.linspace(0, 1, 1000)
    d_wash_line = washout_dilution(alpha_line, params)
    #ax.plot(alpha_line, d_wash_line, color="tab:red", linewidth=3, label="washout")
    ##        label=r"$d=(1-\alpha)\mu(s_{\mathrm{in}})$")
    
    # zero contraction contour 
    c0_color = "tab:red"
    c0 = ax.contour(A, D, C, levels=[0.0], colors=c0_color, linewidths=1)#, linestyles=':')
    #ax.plot([], [], label=r"$c^\star(\alpha,d)=0$", color=c0_color)
    cbar.add_lines(c0)

    # equilibrium point
    if point is not None:
        alpha_star, d_star = point
        ax.plot(alpha_star, d_star, "wo", mec="black", mew=1.2, ms=7, zorder=10)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$d$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, d_max)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    #ax.grid(True, alpha=0.3)
    #legend()

    fig.tight_layout()
    fig.subplots_adjust(
        top=0.98, bottom=0.17,
        left=0.1, right=0.975,
    )

    plt.savefig(f"{PLOT_DIR}/equilibrium_contraction.pdf")
    return fig, ax



# MAIN =========================================================================

if __name__ == "__main__":
    PLOT_DIR = "plots"
    set_mpl_global(output_dir=PLOT_DIR)

    operational_diagram(point=(0.3, 0.15))