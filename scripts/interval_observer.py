from os.path import dirname, realpath
import sys
dir_path = dirname(realpath(__file__))
sys.path += [dir_path, dirname(dir_path)]

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from fluomark.kinetics import monod
from fluomark.mpl_utils import set_mpl_global, legend
from model import default_parameters, default_x0, dilution_func, allocation_func

# MODEL ========================================================================

def plant_dynamics(t, x, alpha_func, d_func, parameters):
    s, b, f = x
    a = alpha_func(t)
    d = d_func(t)
    s_in = parameters["s_in"]
    gamma = parameters["gamma"]
    beta = parameters["beta"]
    r = monod(s, parameters["mu_max"], parameters["ks"]) * b
    ds = -r/gamma + d*(s_in - s)
    db = (1.0 - a) * r - d*b
    df = a * beta * r - d*f
    return np.array([ds, db, df])

def simulate_run(t, x0, alpha_func, d_func, params):
    ode_func = lambda ti, x: plant_dynamics(ti, x, alpha_func, d_func, params)
    sol = solve_ivp(ode_func, (t[0], t[-1]), x0, t_eval=t, rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(sol.message)

    x_true = sol.y.T
    return x_true

# OBSERVER =====================================================================

def interval_observer(t, f_lo, f_hi, sig0_bounds, xi0_bounds,
                      mu_max_bounds, ks_bounds, d_func, params):
    t = np.asarray(t, dtype=float)
    f_lo, f_hi = np.asarray(f_lo, dtype=float), np.asarray(f_hi, dtype=float)
    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("t must contain at least two increasing times")
    if f_lo.shape != t.shape or f_hi.shape != t.shape or np.any(f_lo > f_hi):
        raise ValueError("Invalid fluorescence bands")
    m_lo, m_hi = mu_max_bounds
    k_lo, k_hi = ks_bounds
    if not (0 < m_lo <= m_hi and 0 < k_lo <= k_hi):
        raise ValueError("Monod parameter intervals must be positive and ordered")
    if sig0_bounds[0] > sig0_bounds[1] or xi0_bounds[0] > xi0_bounds[1]:
        raise ValueError("Initial intervals must be ordered")

    s_in, gamma, beta = params["s_in"], params["gamma"], params["beta"]

    def rhs(ti, x):
        sig_lo, sig_hi, xi_lo, xi_hi = x
        flo = np.interp(ti, t, f_lo)
        fhi = np.interp(ti, t, f_hi)
        d = d_func(ti)
        s_lo = max(0.0, sig_lo - xi_lo / gamma)
        s_hi = max(0.0, sig_hi - xi_hi / gamma)
        b_lo = max(0.0, xi_lo - fhi / beta)
        b_hi = max(0.0, xi_hi - flo / beta)
        r_lo = monod(s_lo, m_lo, k_hi) * b_lo
        r_hi = monod(s_hi, m_hi, k_lo) * b_hi
        return [d * (s_in - sig_lo), d * (s_in - sig_hi),
                r_lo - d * xi_lo, r_hi - d * xi_hi]

    x0 = [*sig0_bounds, *xi0_bounds]
    sol = solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t,
                    rtol=1e-9, atol=1e-11, max_step=0.02)
    if not sol.success:
        raise RuntimeError(sol.message)

    sig_lo, sig_hi, xi_lo, xi_hi = sol.y
    s_lo = np.maximum(0.0, sig_lo - xi_hi / gamma)
    s_hi = np.maximum(0.0, sig_hi - xi_lo / gamma)
    b_lo = np.maximum(0.0, xi_lo - f_hi / beta)
    b_hi = np.maximum(0.0, xi_hi - f_lo / beta)
    mu_lo = monod(s_lo, m_lo, k_hi)
    mu_hi = monod(s_hi, m_hi, k_lo)
    r_lo, r_hi = mu_lo * b_lo, mu_hi * b_hi
    return dict(sig_lo=sig_lo, sig_hi=sig_hi, xi_lo=xi_lo, xi_hi=xi_hi,
                s_lo=s_lo, s_hi=s_hi, b_lo=b_lo, b_hi=b_hi,
                mu_lo=mu_lo, mu_hi=mu_hi, r_lo=r_lo, r_hi=r_hi)

# PLOTS ========================================================================


def plot_results(t, x_true, f_lo, f_hi, intervals, alpha_func, params,
                 output_file=None):
    s, b, f = x_true.T
    xi = b + f / params["beta"]
    sig = s + xi / params["gamma"]
    mu = monod(s, params["mu_max"], params["ks"])
    r = mu * b

    quantities = [
        ("s", s, r"$s$ (substrate)"),
        ("sig", sig, r"$\sigma$ (total coordinate)"),
        ("b", b, r"$b$ (bacteria)"),
        ("xi", xi, r"$\xi$ (combined coordinate)"),
        ("f", f, r"$f$ (fluorescence)"),
        ("r", r, r"$r=\mu(s)b$ (total growth rate)"),
        ("alpha", alpha_func(t), r"$\alpha$ (true allocation)"),
        ("mu", mu, r"$\mu(s)$ (specific growth rate)"),
    ]

    fig, ax = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    for a, (key, true, ylabel) in zip(ax.flat, quantities):
        a.plot(t, true, "k", lw=1.4, label="True")
        if key == "f":
            lo, hi = f_lo, f_hi
        elif key == "alpha":
            lo, hi = None, None
        else:
            lo, hi = intervals[f"{key}_lo"], intervals[f"{key}_hi"]
        if lo is not None:
            a.fill_between(t, lo, hi, color="tab:blue", alpha=0.22,
                           label="Interval")
            a.plot(t, lo, color="tab:blue", lw=0.75)
            a.plot(t, hi, color="tab:blue", lw=0.75)
        a.set_ylabel(ylabel)
        a.grid(alpha=0.2)

    #ax[0, 0].set_ylim(0, 0.01)

    legend(ax[0, 0])
    for a in ax[-1]:
        a.set_xlabel(r"$t$ [hours]")
    ax[0, 0].set_xlim(t[0], t[-1])
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file)
    return fig, ax


def plot_results_ecc(t, x_true, f_lo, f_hi, intervals, alpha_func, params,
                 output_file=None):
    s, b, f = x_true.T
    xi = b + f / params["beta"]
    sig = s + xi / params["gamma"]
    mu = monod(s, params["mu_max"], params["ks"])

    quantities = [
        ("s", s, "$s$"),
        ("b", b, "$b$"),
        #("f", f, "$f$"),
        #("sig", sig, r"$\sigma$"),
        #("xi", xi, r"$\xi$"),
    ]

    fig, ax = plt.subplots(2, 1, figsize=(3.5, 2.7), sharex=True)
    for a, (key, true, ylabel) in zip(ax.flat, quantities):
        a.plot(t, true, "k", lw=1.4, label="True")
        if key == "f":
            lo, hi = f_lo, f_hi
        elif key == "alpha":
            lo, hi = None, None
        else:
            lo, hi = intervals[f"{key}_lo"], intervals[f"{key}_hi"]
        if lo is not None:
            a.fill_between(t, lo, hi, facecolor="tab:blue", alpha=0.2, label="Interval")
            #a.plot(t, lo, color="tab:blue", lw=0.75)
            #a.plot(t, hi, color="tab:blue", lw=0.75)
        a.set_ylabel(ylabel)
        a.grid(alpha=0.2)

    ax[1].set_ylim(0, 1.5)


    #legend(ax[0, 0])
    ax[-1].set_xlabel(r"$t$ [hours]")
    ax[-1].set_xlim(t[0], t[-1])
    ax[-1].set_xticks(np.arange(0, t[-1] + 1, 12))
    fig.tight_layout()
    fig.align_labels()
    fig.subplots_adjust(
        left=0.15, right=0.96,
        top=0.98, bottom=0.16,
        hspace=0.12,
    )
    if output_file is not None:
        fig.savefig(output_file)
    return fig, ax


def print_containment(t, x_true, intervals, params):
    s, b, f = x_true.T
    xi = b + f / params["beta"]
    sig = s + xi / params["gamma"]
    mu = monod(s, params["mu_max"], params["ks"])
    truth = {"sig": sig, "xi": xi, "s": s, "b": b,
             "mu": mu, "r": mu * b}
    for key, actual in truth.items():
        lo, hi = intervals[f"{key}_lo"], intervals[f"{key}_hi"]
        violation = max(np.max(lo - actual), np.max(actual - hi), 0.0)
        print(f"{key:>3s}: max sampled containment violation = {violation:.3e}")


# MAIN =========================================================================

if __name__ == "__main__":
    PLOT_DIR = "plots"
    set_mpl_global(output_dir=PLOT_DIR)

    # model parameters
    params = default_parameters
    x0 = default_x0 # np.array([0.0, 1, 0.0])
    mu_max_bounds = (0.5, 9.0)
    ks_bounds = (0.001, 0.005)
    
    # state bounds
    f_half_band = 0.02
    s0_lo, s0_hi = 0, 0.3
    b0_lo, b0_hi = 0.85, 1.15
    
    # alpha(t): logistic function 0 -> alpha_max
    alpha_func = lambda t: allocation_func(t, t_switch=24, alpha_max=0.3, r=0.25)

    # d(t): constant dilution rate
    # 0.15 per hour in Judith's experiment
    t_switch = 3.0
    d_func = lambda t: dilution_func(t, t_switch=t_switch, d0=1.3, d_star=0.15)

    # time steps
    dt = 1   # minutes
    tf = 60  # hours
    t = np.arange(0, tf + dt / 60, dt / 60)

    # simulate model
    x_true = simulate_run(t, x0, alpha_func, d_func, params)
    f_true = x_true[:, 2]
    f_lo, f_hi = np.maximum(0, f_true - f_half_band), f_true + f_half_band
    
    # initial estimates
    s0, b0, f0 = x0
    xi0 = b0 + f0 / params["beta"]
    sig0 = s0 + xi0 / params["gamma"]

    # get bounds from x0 bounds
    xi0_lo = b0_lo + f_lo[0]/params['beta']
    xi0_hi = b0_hi + f_hi[0]/params['beta']
    sig0_lo = s0_lo + xi0_lo/params['gamma']
    sig0_hi = s0_hi + xi0_hi/params['gamma']
    print('Initial bounds:')
    print(f"s0 in [{s0_lo, s0_hi}]")
    print(f"b0 in [{b0_lo, b0_hi}]")
    print(f"f0 in [{f_lo[0], f_hi[0]}]")
    print(f"sigma0 in [{sig0_lo, sig0_hi}] => width: {sig0_hi - sig0_lo}")
    print(f"xi0 in [{xi0_lo, xi0_hi}] => width: {xi0_hi - xi0_lo}")
    sig0_bounds = sig0_lo, sig0_hi
    xi0_bounds = xi0_lo, xi0_hi

    intervals = interval_observer(t, f_lo, f_hi, sig0_bounds, xi0_bounds,
                                  mu_max_bounds, ks_bounds, d_func, params)
    print("Monod parameter intervals:", mu_max_bounds, ks_bounds)
    #print_containment(t, x_true, intervals, params)
    plot_results(t, x_true, f_lo, f_hi, intervals, alpha_func, params,
                 output_file=f"{PLOT_DIR}/interval_observer.pdf")
    plot_results_ecc(t, x_true, f_lo, f_hi, intervals, alpha_func, params,
                     output_file=f"{PLOT_DIR}/interval_observer_ecc.pdf")
