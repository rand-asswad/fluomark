from os.path import dirname, realpath
import sys
dir_path = dirname(realpath(__file__))
sys.path += [dir_path, dirname(dir_path)]

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from fluomark.kinetics import monod
from fluomark.kalman import calibrate_kappa, kalman_filter
from fluomark.mpl_utils import set_mpl_global, legend

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

def simulate_noisy_run(t, x0, alpha_func, d_func, params, rng):
    ode_func = lambda ti, x: plant_dynamics(ti, x, alpha_func, d_func, params)
    sol = solve_ivp(ode_func, (t[0], t[-1]), x0, t_eval=t, rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(sol.message)

    x_true = sol.y.T
    y = np.maximum(x_true[:, 2] + rng.normal(0, params["noise_std"], len(t)), 0)
    return x_true, y

# OBSERVER =====================================================================

def observer_dynamics(t, z_hat, f_hat, d_func, parameters):
    sigma, xi = z_hat
    d = d_func(t)
    s_in = parameters["s_in"]
    s = sigma - xi / parameters["gamma"]
    b = xi - f_hat / parameters["beta"]
    r = monod(s, parameters["mu_max"], parameters["ks"]) * b
    dsigma = d * (s_in - sigma)
    dxi = r - d * xi
    return np.array([dsigma, dxi])

def estimate_alpha(t, f, fdot, s, b, d_func, params):
    r = monod(s, params["mu_max"], params["ks"]) * b
    denominator = params["beta"] * r
    d = np.array([d_func(ti) for ti in t])
    numerator = fdot + d * f
    return np.divide(numerator, denominator, out=np.full_like(f, np.nan),
                     where=np.abs(denominator) > 1e-10)

def run_online_estimator(t, y, z0_hat, kappa, d_func, params):
    filtered = kalman_filter(t, y, kappa, params["noise_std"]**2)
    f_samples, fdot_samples = filtered.states.T

    z_hat = np.array(z0_hat, dtype=float)
    sig_samples = np.empty(len(t))
    xi_samples = np.empty(len(t))
    sig_samples[0], xi_samples[0] = z_hat
    continuous = {key: [] for key in ("t", "f", "fdot", "sig", "xi")}

    for k in range(len(t) - 1):
        t_k, t_next = t[k], t[k + 1]
        f_k, fdot_k = filtered.states[k]

        def f_prediction(ti):
            return f_k + (ti - t_k) * fdot_k

        def observer_rhs(ti, x):
            return observer_dynamics(ti, x, f_prediction(ti), d_func, params)

        sol = solve_ivp(observer_rhs, (t_k, t_next), z_hat,
                        dense_output=True, rtol=1e-7, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"Observer integration failed: {sol.message}")

        ti = np.linspace(t_k, t_next, 30, endpoint=False)
        sig_i, xi_i = sol.sol(ti)
        continuous["t"].append(ti)
        continuous["f"].append(f_prediction(ti))
        continuous["fdot"].append(np.full_like(ti, fdot_k))
        continuous["sig"].append(sig_i)
        continuous["xi"].append(xi_i)

        z_hat = sol.y[:, -1]
        sig_samples[k + 1], xi_samples[k + 1] = z_hat

    continuous = {key: np.concatenate(value) for key, value in continuous.items()}
    samples = {"f": f_samples, "fdot": fdot_samples,
               "sig": sig_samples, "xi": xi_samples}

    for estimates, times in ((continuous, continuous["t"]), (samples, t)):
        estimates["s"] = estimates["sig"] - estimates["xi"] / params["gamma"]
        estimates["b"] = estimates["xi"] - estimates["f"] / params["beta"]
        estimates["alpha"] = estimate_alpha(
            times, estimates["f"], estimates["fdot"], estimates["s"],
            estimates["b"], d_func, params)

    samples["alpha"][0] = np.nan
    return continuous, samples

# PLOTS ========================================================================

def plot_results(t, y, x_true, continuous, samples, alpha_func, params,
                 output_file=None):
    s_true, b_true, f_true = x_true.T
    alpha_true = alpha_func(t)
    xi_true = b_true + f_true / params["beta"]
    sig_true = s_true + xi_true / params["gamma"]
    mu_true = monod(s_true, params["mu_max"], params["ks"])

    fig, ax = plt.subplots(4, 2, figsize=(14, 12), sharex=True)

    ax[0, 0].plot(t, s_true, "--", label="$s$")
    ax[0, 0].plot(t, samples["s"], label=r"$\hat{s}=\hat{\sigma}-\hat{\xi}/\gamma$")
    ax[0, 0].set_ylabel(r"$s$")
    ax[0, 0].set_ylim(0, 0.003)

    ax[1, 0].plot(t, b_true, "--", label="$b$")
    ax[1, 0].plot(t, samples["b"], label=r"$\hat{b}=\hat{\xi}-\hat{f}/\beta$")
    ax[1, 0].set_ylabel(r"$b$")

    ax[2, 0].plot(t, y, "C1.", alpha=0.25, label="Noisy measurements")
    ax[2, 0].plot(t, f_true, "C0--", label="$f$")
    ax[2, 0].plot(continuous["t"], continuous["f"], label=r"$\hat{f}$")
    ax[2, 0].set_ylabel(r"$f$")

    ax[3, 0].plot(t, alpha_true, "--", label=r"$\alpha$")
    ax[3, 0].plot(continuous["t"], continuous["alpha"], label=r"$\hat{\alpha}$")
    ax[3, 0].set_ylabel(r"$\alpha$")
    ax[3, 0].set_xlabel(r"$t$ (hours)")
    ax[3, 0].set_ylim(-0.3, 1.3)

    ax[0, 1].plot(t, sig_true, "--", label=r"$\sigma$")
    ax[0, 1].plot(t, samples["sig"], label=r"$\hat{\sigma}$")
    ax[0, 1].set_ylabel(r"$\sigma$")

    ax[1, 1].plot(t, xi_true, "--", label=r"$\xi$")
    ax[1, 1].plot(t, samples["xi"], label=r"$\hat{\xi}$")
    ax[1, 1].set_ylabel(r"$\xi$")

    ax[2, 1].plot(t, mu_true, "--", label=r"$\mu(s)$")
    mu_hat = monod(samples["s"], params["mu_max"], params["ks"])
    ax[2, 1].plot(t, mu_hat, label=r"$\mu(\hat{s})$")
    ax[2, 1].set_ylabel(r"$\mu$")

    ax[3, 1].plot(continuous["t"], continuous["fdot"],
                  label=r"$\widehat{\dot{f}}$")
    ax[3, 1].set_ylabel(r"$\dot{f}$")
    ax[3, 1].set_xlabel(r"$t$ (hours)")

    for a in ax.flat:
        a.grid(True)
        legend(a)
    ax[0, 0].set_xlim(t[0], t[-1])

    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file)
    return fig, ax

def plot_results_ecc(t, y, x_true, continuous, samples, alpha_func, d_func,
                     params, output_file=None):
    s_true, b_true, f_true = x_true.T
    alpha_true = alpha_func(t)
    fdot_true = np.array([
        plant_dynamics(ti, xi, alpha_func, d_func, params)[2]
        for ti, xi in zip(t, x_true)
    ])
    xi_true = b_true + f_true / params["beta"]
    sig_true = s_true + xi_true / params["gamma"]
    mu_true = monod(s_true, params["mu_max"], params["ks"])
    mu_hat = monod(samples["s"], params["mu_max"], params["ks"])

    color_left  = "tab:blue" #"C0"
    color_right = "tab:red"  #"C1"
    color_meas  = "#39FF14"

    fig, ax = plt.subplots(4, 1, figsize=(4.0, 4.0), sharex=True)
    axr = [a.twinx() for a in ax]

    def plot_pair(i, left_true, left_hat, right_true, right_hat,
                  ylabel_left, ylabel_right, t_left=None, t_right=None):
        tl = t if t_left is None else t_left
        tr = t if t_right is None else t_right

        ax[i].plot(t, left_true, "--", color=color_left, lw=1.5)
        ax[i].plot(tl, left_hat, color=color_left, lw=1.0)
        axr[i].plot(t, right_true, "--", color=color_right, lw=1.5)
        axr[i].plot(tr, right_hat, color=color_right, lw=1.0)

        ax[i].set_ylabel(ylabel_left, color=color_left, labelpad=3)
        axr[i].set_ylabel(ylabel_right, color=color_right, labelpad=3)
        ax[i].tick_params(axis="y", colors=color_left, labelsize=7, pad=2)
        axr[i].tick_params(axis="y", colors=color_right, labelsize=7, pad=2)
        ax[i].spines["left"].set_color(color_left)
        axr[i].spines["right"].set_color(color_right)

        ax[i].grid(axis="y", alpha=0.2, lw=0.5)
        ax[i].text(0.02, 0.95, f"({chr(97 + i)})",
                   transform=ax[i].transAxes, ha="left", va="top", fontsize=11)

    plot_pair(0, f_true, continuous["f"], fdot_true, continuous["fdot"],
              r"$f$", r"$\dot{f}$", continuous["t"], continuous["t"])
    ax[0].scatter(t, y, s=3, color=color_meas, alpha=0.25,
                  edgecolors="none", zorder=2)

    plot_pair(1, sig_true, samples["sig"], xi_true, samples["xi"],
              r"$\sigma$", r"$\xi$")

    plot_pair(2, s_true, samples["s"], b_true, samples["b"],
              r"$s$", r"$b$")

    plot_pair(3, alpha_true, continuous["alpha"], mu_true, mu_hat,
              r"$\alpha$", r"$\mu$", continuous["t"])

    handles = [
        Line2D([0], [0], color="0.25", lw=1.5, ls="--", label="True"),
        Line2D([0], [0], color="0.25", lw=1.0, label="Estimated"),
        Line2D([0], [0], marker=".", color=color_meas, lw=0,
               markersize=5, label="Measurements")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9.0, bbox_to_anchor=(0.5, 1.05),
               handlelength=2.5, columnspacing=1.3)

    ax[-1].set_xlabel(r"$t$ [hours]", fontsize=9)
    ax[-1].set_xticks(np.arange(0, t[-1]+1, 12))
    ax[-1].tick_params(axis="x", labelsize=8)
    ax[0].set_xlim(t[0], t[-1])
    
    
    ax[0].set_ylim(-5e-3, 74e-3)
    axr[0].set_ylim(-5e-3, 7e-3)
    ax[1].set_ylim(0, 0.55)
    axr[1].set_ylim(0, 0.3)
    ax[2].set_ylim(0.5e-3, 2.8e-3)
    #axr[2].set_ylim(0, 0.3)
    ax[3].set_ylim(-0.01, 0.7)
    axr[3].set_ylim(0.1, 0.7)


    for a in [ax[0], axr[0], ax[2]]:
        a.ticklabel_format(axis='y', scilimits=(-3, -3))
        a.yaxis.get_offset_text().set_fontsize(5)

    fig.align_ylabels()
    fig.subplots_adjust(left=-0.01, right=1.01, bottom=0.04,
                        top=0.95, hspace=0.20)

    if output_file is not None:
        fig.savefig(output_file, bbox_inches="tight")
    return fig, ax, axr



def plot_results_ecc_alt(t, y, x_true, continuous, samples, alpha_func,
                         d_func, params, output_file=None):
    s_true, b_true, f_true = x_true.T
    alpha_true = alpha_func(t)
    fdot_true = np.array([
        plant_dynamics(ti, xi, alpha_func, d_func, params)[2]
        for ti, xi in zip(t, x_true)
    ])
    xi_true = b_true + f_true / params["beta"]
    sig_true = s_true + xi_true / params["gamma"]
    mu_true = monod(s_true, params["mu_max"], params["ks"])
    mu_hat = monod(samples["s"], params["mu_max"], params["ks"])

    color_left = "tab:blue"
    color_right = "tab:red"
    color_meas = "#39FF14"

    def make_figure():
        fig, ax = plt.subplots(2, 1, figsize=(3.5, 3.1), sharex=True)
        axr = [a.twinx() for a in ax]
        return fig, ax, axr

    def plot_pair(a, ar, left_true, left_hat, right_true, right_hat,
                  ylabel_left, ylabel_right, t_left=None, t_right=None):
        tl = t if t_left is None else t_left
        tr = t if t_right is None else t_right

        a.plot(t, left_true, "--", color=color_left, lw=2.0)
        a.plot(tl, left_hat, color=color_left, lw=1.3)
        ar.plot(t, right_true, "--", color=color_right, lw=2.0)
        ar.plot(tr, right_hat, color=color_right, lw=1.3)

        a.set_ylabel(ylabel_left, color=color_left, fontsize=12, labelpad=2)
        ar.set_ylabel(ylabel_right, color=color_right, fontsize=12, labelpad=2)
        a.tick_params(axis="y", colors=color_left, labelsize=10, pad=2)
        ar.tick_params(axis="y", colors=color_right, labelsize=10, pad=2)
        a.spines["left"].set_color(color_left)
        ar.spines["right"].set_color(color_right)
        a.grid(axis="y", alpha=0.15, lw=0.5)

    def sci_axis(a, side="left"):
        a.ticklabel_format(axis="y", scilimits=(-3, -3))
        a.yaxis.get_offset_text().set_visible(False)
        x = 0.01 if side == "left" else 0.99
        color = color_left if side == "left" else color_right
        a.text(x, 0.98, r"$\times 10^{-3}$", transform=a.transAxes,
               color=color, fontsize=9, ha=side, va="top")

    def finish_figure(fig, ax, filename, measurements=True):
        ax[-1].set_xlabel(r"$t$ [hours]", fontsize=11, labelpad=2)
        ax[-1].set_xticks(np.arange(0, t[-1] + 1, 12))
        ax[-1].tick_params(axis="x", labelsize=10, pad=2)
        ax[0].set_xlim(t[0], t[-1])

        handles = [
            Line2D([0], [0], color="0.25", lw=2, ls="--", label="True"),
            Line2D([0], [0], color="0.25", lw=1.3, label="Estimated"),
        ]
        if measurements:
            handles += [
                Line2D([0], [0], marker=".", color=color_meas, lw=0,
                    markersize=6, label="Measurements")
            ]

        fig.legend(handles=handles, loc="upper center", ncol=3,
                frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02),
                handlelength=1.8, columnspacing=0.8)

        fig.subplots_adjust(left=0.13, right=0.85, bottom=0.12,
                            top=0.92, hspace=0.12)
        fig.align_ylabels()

        if filename is not None:
            fig.savefig(filename)

    fig1, ax1, axr1 = make_figure()

    plot_pair(ax1[0], axr1[0], f_true, continuous["f"],
              fdot_true, continuous["fdot"], r"$f$", r"$\dot{f}$",
              continuous["t"], continuous["t"])
    ax1[0].scatter(t, y, s=5, color=color_meas, alpha=0.3,
                   edgecolors="none", zorder=2)

    plot_pair(ax1[1], axr1[1], sig_true, samples["sig"],
              xi_true, samples["xi"], r"$\sigma$", r"$\xi$")

    ax1[0].set_ylim(-5e-3, 74e-3)
    axr1[0].set_ylim(-5e-3, 7e-3)
    ax1[1].set_ylim(0, 0.55)
    axr1[1].set_ylim(0, 0.3)

    sci_axis(axr1[0], "right")
    finish_figure(fig1, ax1, f"{output_file}_1.pdf" if output_file else None,
                  measurements=True)

    fig2, ax2, axr2 = make_figure()

    plot_pair(ax2[0], axr2[0], s_true, samples["s"],
              b_true, samples["b"], r"$s$", r"$b$")

    plot_pair(ax2[1], axr2[1], alpha_true, continuous["alpha"],
              mu_true, mu_hat, r"$\alpha$", r"$\mu$",
              continuous["t"])

    ax2[0].set_ylim(0.5e-3, 2.8e-3)
    ax2[1].set_ylim(-0.01, 0.7)
    axr2[1].set_ylim(0.1, 0.7)

    sci_axis(ax2[0], "left")
    finish_figure(fig2, ax2, f"{output_file}_2.pdf" if output_file else None,
                  measurements=False)
    #plt.show()

    return (fig1, ax1, axr1), (fig2, ax2, axr2)



# MAIN =========================================================================

if __name__ == "__main__":
    PLOT_DIR = "plots"
    set_mpl_global(output_dir=PLOT_DIR)

    # model parameters
    params = {
        "s_in": 0.5,
        "beta": 0.9,
        "gamma": 0.45,
        "mu_max": 0.8,
        "ks": 0.002,
        "noise_std": 0.005,
    }
    x0 = np.array([0.01, 0.15, 0.0])
    x0_hat = np.array([0.001, 0.05, 0.0])

    # alpha(t): logistic function 0 -> alpha_max
    alpha_max = 0.3
    r = 0.25
    t_switch = 24
    alpha_func = lambda t: alpha_max / (1.0 + np.exp(-r * (t - t_switch)))

    # d(t): constant dilution rate
    d_func = lambda t: 0.2
    
    # time steps
    dt = 5   # minutes
    tf = 72  # hours
    t = np.arange(0, tf + dt / 60, dt / 60)

    # simulate model
    simulation_seed = 42
    rng = np.random.default_rng(simulation_seed)
    x_true, y = simulate_noisy_run(t, x0, alpha_func, d_func, params, rng)
    
    # calibrate kappa
    calibration_seed = 5420627
    rng = np.random.default_rng(calibration_seed)
    _, y_cal = simulate_noisy_run(t, x0, alpha_func, d_func, params, rng)
    kappa_hat, log_likelihood = calibrate_kappa(t, y_cal, params["noise_std"]**2)
    print(f"Estimated kappa:\t\t{kappa_hat:.6e}")
    print(f"Calibration log-likelihood:\t{log_likelihood:.6f}")
    
    
    s0_hat, b0_hat, f0_hat = x0_hat
    xi0_hat = b0_hat + f0_hat / params["beta"]
    sig0_hat = s0_hat + xi0_hat / params["gamma"]
    z0_hat = np.array([sig0_hat, xi0_hat])

    continuous, samples = run_online_estimator(t, y, z0_hat, kappa_hat, d_func, params)
    plot_results(t, y, x_true, continuous, samples, alpha_func, params,
                output_file=f"{PLOT_DIR}/observer_pipeline_full.pdf")

    #plot_results_ecc(t, y, x_true, continuous, samples, alpha_func, d_func,
    #            params, output_file=f"{PLOT_DIR}/observer_pipeline_ecc.pdf")

    plot_results_ecc_alt(t, y, x_true, continuous, samples, alpha_func,
                d_func, params, output_file=f"{PLOT_DIR}/observer_pipeline")

