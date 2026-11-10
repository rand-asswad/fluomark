from os.path import dirname, realpath
import sys
dir_path = dirname(realpath(__file__))
sys.path += [dir_path, dirname(dir_path)]

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from fluomark.kinetics import monod, monod_derivative
from fluomark.kalman import calibrate_kappa, kalman_filter
from fluomark.mpl_utils import set_mpl_global, legend, add_data_inset
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
    #ax[0, 0].set_ylim(0, 0.003)

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
                     params, t_switch, output_file=None):
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

    color_true = "tab:blue"
    color_hat = "tab:red"
    color_meas = "#39FF14"
    color_switch = "0.55"

    # Figure: 4 rows x 2 columns
    fig, ax = plt.subplots(4, 2, figsize=(7.15, 5), sharex=True)

    def plot_variable(a, true, estimated, ylabel, t_est=None):
        if t_est is None:
            t_est = t

        a.axvline(t_switch, color=color_switch, ls="--", lw=1, zorder=1)
        a.plot(t, true, color=color_true, lw=1.5)
        a.plot(t_est, estimated, color=color_hat, lw=1.5)

        a.set_ylabel(ylabel, fontsize=11)
        a.tick_params(axis="both", labelsize=8)
        a.grid(axis="y", alpha=0.2, lw=0.5)

    # Left column: f, fdot, sigma, xi
    plot_variable(ax[0, 0], f_true, continuous["f"], "$f$", continuous["t"])
    plot_variable(ax[1, 0], fdot_true, continuous["fdot"], r"$\dot{f}$", continuous["t"])
    plot_variable(ax[2, 0], sig_true, samples["sig"], r"$\sigma$")
    plot_variable(ax[3, 0], xi_true, samples["xi"], r"$\xi$")

    ax[0, 0].scatter(t, y, s=5, color=color_meas, alpha=0.35,
                     edgecolors="none", zorder=2)

    # Right column: s, b, alpha, mu
    plot_variable(ax[0, 1], s_true, samples["s"], "$s$")
    plot_variable(ax[1, 1], b_true, samples["b"], "$b$")
    plot_variable(ax[2, 1], alpha_true, continuous["alpha"], r"$\alpha$", continuous["t"])
    plot_variable(ax[3, 1], mu_true, mu_hat, r"$\mu$")

    # Shared legend
    handles = [
        Line2D([0], [0], color=color_true, lw=1.5, label="True"),
        Line2D([0], [0], color=color_hat, lw=1.0, label="Estimated"),
        Line2D([0], [0], marker=".", color=color_meas, lw=0,
               markersize=5, label="Measurements")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.05),
               handlelength=2.5, columnspacing=1.3)

    t_zoom = (36, 48)
    zoom_position = (0.62, 0.08, 0.35, 0.50)
    zoom_f = add_data_inset(ax[0, 0], t_zoom, zoom_position, noticks=True, visible=(0, 3))

    # Axes formatting
    for a in ax.flat:
        a.set_xlim(t[0], t[-1])
        a.tick_params(axis="both", labelsize=8)

    for a in ax[-1, :]:
        a.set_xlabel(r"$t$ [hours]", fontsize=10)
        a.set_xticks(np.arange(0, t[-1] + 1, 12))

    for a in [ax[1,0]]:
        a.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        a.yaxis.get_offset_text().set_fontsize(5)

    ax[1,0].set_ylim(-0.5e-2, 1.5e-2) # df/dt
    ax[2,0].set_ylim(1.5, 3.5) # sigma
    ax[3,0].set_ylim(0.1, 1.4) # xi
    ax[1,1].set_ylim(0.1, 1.4) # b
    ax[2,1].set_ylim(-0.02, 0.35) # alpha
    ax[3,1].set_ylim(0.1, 0.75) # alpha

    fig.align_ylabels()
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.07,
                        top=0.99, hspace=0.15, wspace=0.18)

    if output_file is not None:
        fig.savefig(output_file, bbox_inches="tight")
    return fig, ax

def plot_contraction_diagnostics(t, x_true, continuous, samples, d_func,
                                 params, t_switch, lambdas=(0.1, 0.5, 1.0),
                                 output_file=None):
    t = np.asarray(t)
    s_true, b_true, f_true = x_true.T

    gamma = params["gamma"]
    beta = params["beta"]
    mu = lambda s: monod(s, params['mu_max'], params['ks'])
    mu_deriv = lambda s: monod_derivative(s, params['mu_max'], params['ks'])
    d = np.array([d_func(ti) for ti in t])

    xi_true = b_true + f_true / beta
    sig_true = s_true + xi_true / gamma
    f_hat = np.interp(t, continuous["t"], continuous["f"])

    z_true = np.column_stack((sig_true, xi_true, f_true))
    z_hat = np.column_stack((samples["sig"], samples["xi"], f_hat))

    # Local Jacobian along the true trajectory.
    c_true = mu(s_true) - d - mu_deriv(s_true) * b_true / gamma

    # Segment-averaged Jacobian, including nonzero fluorescence error.
    # This matches the positive-part extension of the growth term.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    nodes = (nodes + 1) / 2
    weights = weights / 2
    A = np.zeros_like(t, dtype=float)

    for theta, weight in zip(nodes, weights):
        z_theta = z_true + theta * (z_hat - z_true)
        s_raw = z_theta[:, 0] - z_theta[:, 1] / gamma
        b_raw = z_theta[:, 1] - z_theta[:, 2] / beta
        s_theta = np.maximum(s_raw, 0.0)
        b_theta = np.maximum(b_raw, 0.0)
        c_theta = (mu(s_theta) * (b_raw > 0)
                   - mu_deriv(s_theta) * b_theta * (s_raw > 0) / gamma - d)
        A += weight * c_theta

    F_A = cumulative_trapezoid(A, t, initial=0.0)
    F_c = cumulative_trapezoid(c_true, t, initial=0.0)

    M_curves = {}
    for lam in lambdas:
        H = F_A + lam * (t - t[0])
        running_min = np.minimum.accumulate(H)
        log_M = np.maximum.accumulate(H - running_min)
        M_curves[lam] = np.exp(np.minimum(log_M, 700))

    #fig, ax = plt.subplots(3, 1, figsize=(7.15, 6.0), sharex=True)

    #ax[0].plot(t, c_true, color="tab:blue", lw=1.3,
    #           label=r"$c(z(t),d(t))$")
    #ax[0].plot(t, A, color="tab:red", lw=1.3, label=r"$A(t)$")
    #ax[0].set_ylabel(r"$c,\ A$ [h$^{-1}$]")
    #ax[0].legend(frameon=False, fontsize=9)

    #ax[1].plot(t, F_c, color="tab:blue", lw=1.3,
    #           label=r"$\int_0^t c(z(u),d(u))\,du$")
    #ax[1].plot(t, F_A, color="tab:red", lw=1.3,
    #           label=r"$\int_0^t A(u)\,du$")
    #ax[1].set_ylabel("Cumulative integral")
    #ax[1].legend(frameon=False, fontsize=9)

    #for lam, M in M_curves.items():
    #    ax[2].plot(t, M, lw=1.3, label=rf"$\lambda={lam:g}$")
    #ax[2].set_ylabel(r"$M_\lambda(T)$")
    #ax[2].set_xlabel(r"$T$ [hours]")
    #ax[2].legend(frameon=False, fontsize=9)

    #for a in ax:
    #    a.axvline(t_switch, color="0.55", ls="--", lw=1)
    #    a.grid(axis="y", alpha=0.2, lw=0.5)
    #    a.tick_params(axis="both", labelsize=8)
    #    a.set_xlim(t[0], t[-1])

    #ax[0].axhline(0, color="0.4", lw=0.7)
    #ax[1].axhline(0, color="0.4", lw=0.7)
    #fig.align_ylabels()
    #fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09,
    #                    top=0.98, hspace=0.18)

    fig, ax = plt.subplots(2, 1, figsize=(5, 2.5), sharex=True)
    ax_int = ax[0].twinx()

    # Local contraction coefficient and accumulated contraction
    lc, = ax[0].plot(t, c_true, color="tab:blue", lw=1.3, label=r"$c(z(t),d(t))$")
    la, = ax_int.plot(t, F_A, color="tab:red", lw=1.3, label=r"$\int_0^t A(u)\,du$")

    #ax[0].set_ylabel(r"$c(z(t),d(t))$ [h$^{-1}$]", color="tab:blue")
    #ax_int.set_ylabel(r"$\int_0^t A(u)\,du$", color="tab:red")
    ax[0].tick_params(axis="y", colors="tab:blue")
    ax_int.tick_params(axis="y", colors="tab:red")
    ax[0].legend(handles=[lc, la], loc="upper right", frameon=False,
                 fontsize=13, ncols=2, bbox_to_anchor=(1.0, 1.0),
                 borderpad=0, labelspacing=0.1, borderaxespad=0.3)

    # Minimum accumulated-contraction constant
    for lam, M in M_curves.items():
        ax[1].plot(t, np.log(M), lw=1.3, label=rf"$\lambda={lam:g}$")

    ax[1].set_ylabel(r"$\ln M_\lambda(t)$")
    ax[1].set_xlabel(r"$t$ [hours]")
    ax[1].legend(frameon=False, fontsize=12, loc="upper right", ncols=3,
                 bbox_to_anchor=(1.0, 0.88), borderpad=0)

    for a in ax:
        a.axvline(t_switch, color="0.55", ls="--", lw=1)
        a.grid(axis="y", alpha=0.2, lw=0.5)
        a.tick_params(axis="both", labelsize=8)
        a.set_xlim(t[0], t[-1])
    ax[0].set_ylim(ax[0].get_ylim()[0], 70)
    ax_int.set_ylim(ax_int.get_ylim()[0], 2900)
    ax[1].set_ylim(ax[1].get_ylim()[0], 5.4)
    
    ax_int.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
    ax_int.yaxis.get_offset_text().set_fontsize(9)

    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.2,
                        top=0.945, hspace=0.1)

    if output_file is not None:
        fig.savefig(output_file, bbox_inches="tight")
    #plt.show()

    return fig, ax, {
        "c": c_true,
        "A": A,
        "F_c": F_c,
        "F_A": F_A,
        "M": M_curves,
        "e_f": f_hat - f_true,
    }


# MAIN =========================================================================

if __name__ == "__main__":
    PLOT_DIR = "plots"
    set_mpl_global(output_dir=PLOT_DIR)

    # model parameters
    params = default_parameters
    x0 = default_x0 # np.array([0.0, 1, 0.0])
    x0_hat = np.array([0.5, 1.5, 0.0])

    # alpha(t): logistic function 0 -> alpha_max
    alpha_func = lambda t: allocation_func(t, t_switch=24, alpha_max=0.3, r=0.25)

    # d(t): constant dilution rate
    # 0.15 per hour in Judith's experiment
    t_switch = 3.0
    d_func = lambda t: dilution_func(t, t_switch=t_switch, d0=1.3, d_star=0.15)
    
    # time steps
    dt = 10  # minutes
    tf = 60  # hours
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
    
    # initial estimates
    s0_hat, b0_hat, f0_hat = x0_hat
    xi0_hat = b0_hat + f0_hat / params["beta"]
    sig0_hat = s0_hat + xi0_hat / params["gamma"]
    z0_hat = np.array([sig0_hat, xi0_hat])

    continuous, samples = run_online_estimator(t, y, z0_hat, kappa_hat, d_func, params)
    plot_results(t, y, x_true, continuous, samples, alpha_func, params,
        output_file=f"{PLOT_DIR}/observer_pipeline_full.pdf")

    plot_results_ecc(t, y, x_true, continuous, samples, alpha_func, d_func,
        params, t_switch, output_file=f"{PLOT_DIR}/observer_pipeline_ecc.pdf")

    fig, ax, contraction = plot_contraction_diagnostics(
        t, x_true, continuous, samples, d_func, params, t_switch,
        output_file=f"{PLOT_DIR}/contraction_diagnostics.pdf"
    )
