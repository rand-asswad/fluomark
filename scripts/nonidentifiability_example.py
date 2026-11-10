import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from fluomark.kinetics import monod
from fluomark.mpl_utils import set_mpl_global, legend


# MODEL PARAMETERS -------------------------------------------------------------

s_in = 2.0
beta = 0.8
gamma = 0.45

mu_max = 0.7
ks = 0.002
mu = lambda s: monod(s, mu_max/2, ks)

factor = 1/2
mu2 = lambda s: monod(s, mu_max, ks*factor)

def d_func(t):
    return 0.25


# ALLOCATION: USED ONLY BY THE PLANT -------------------------------------------

def alpha_true(t):
    alpha_steady = 0.3
    rate = 0.25
    t_switch = 24.0

    return alpha_steady / (
        1.0 + np.exp(-rate * (t - t_switch))
    )


# PLANT ------------------------------------------------------------------------

def plant1_dynamics(t, x):
    s, b, f = x

    d = d_func(t)
    r = mu(s) * b
    alpha = alpha_true(t)

    ds = -r / gamma + d * (s_in - s)
    db = (1.0 - alpha) * r - d * b
    df = alpha * beta * r - d * f

    return [ds, db, df]

def plant2_dynamics(t, x, q):
    s, b, f = x

    d = d_func(t)
    r = mu2(s) * b

    ds = -r / gamma + d * (s_in - s)
    db = r - q(t) - d * b
    df = beta * q(t) - d * f

    return [ds, db, df]


# MAIN -------------------------------------------------------------------------

if __name__ == "__main__":
    set_mpl_global()

    tf = 60.0
    t_eval = np.linspace(0.0, tf, 2000)

    # True initial conditions
    x0 = np.array([0.1 * s_in, 0.1, 0.0])

    # Simulate plant
    sol1 = solve_ivp(
        plant1_dynamics,
        (0.0, tf),
        x0,
        method="RK45",
        dense_output=True,
        rtol=1e-10,
        atol=1e-12
    )
    def q(t):
        x1 = sol1.sol(t)
        s = x1[0]
        b = x1[1]
        return alpha_true(t)*mu(s)*b


    x1 = sol1.sol(t_eval)
    s1 = x1[0]
    b1 = x1[1]
    f1 = x1[2]
    alpha1 = np.array([alpha_true(ti) for ti in t_eval])
    
    sol2 = solve_ivp(
        lambda t, x: plant2_dynamics(t, x, q),
        (0.0, tf),
        x0,
        method="RK45",
        dense_output=True,
        rtol=1e-10,
        atol=1e-12
    )
    x2 = sol2.sol(t_eval)
    s2 = x2[0]
    b2 = x2[1]
    f2 = x2[2]
    alpha2 = q(t_eval) / (mu2(s2) * b2)

    # PLOT ---------------------------------------------------------------------

    fig, axes = plt.subplots(2, 1, figsize=(5, 2.6), sharex=True)

    # LEFT COLUMN: BIOLOGICAL STATES AND ALLOCATION ----------------------------

    axes[0].plot(t_eval, s1, 'C0', label="$s$")
    axes[0].plot(t_eval, s2, 'C0', ls="--")#, label="$\\bar{s}$")
    axes[0].plot(t_eval, b1, 'C2', label="$b$")
    axes[0].plot(t_eval, b2, 'C2', ls="--")#, label="$\\bar{b}$")
    axes[0].plot(t_eval, f1, 'C1', label="$f$")
    axes[0].legend(ncols=3, loc="upper right", frameon=False,
                   bbox_to_anchor=(1.0, 1.0), borderpad=0)
    
    axes[1].plot(t_eval, alpha1, 'C0', label=r"$\alpha$")
    axes[1].plot(t_eval, alpha2, 'C0', ls="--")#, label=r"$\bar{\alpha}$")
    axes[1].plot(t_eval, mu(s1), 'C1', label=r"$\mu(s)$")
    axes[1].plot(t_eval, mu2(s2), 'C1', ls="--")#, label=r"$\bar{\mu}(\bar{s})$")
    #axes[1].plot(t_eval, alpha1*mu(s1)*b1, 'C2', label=r"$q_1$")
    #axes[1].plot(t_eval, alpha2*mu2(s2)*b2, 'C2', ls="--", label=r"$q_2$")
    axes[1].legend(ncols=2, loc="upper right", frameon=False,
                   bbox_to_anchor=(1.0, 1.0), borderpad=0)


    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].set_xlim(0, tf)
    #axes[0].set_ylim(0, 0.3)
    #axes[1].set_ylim(0, 1.2)
    axes[1].set_xlabel(r"$t$ [hours]")

    fig.tight_layout()
    fig.subplots_adjust(
        top=0.98, bottom=0.175,#0.1,
        left=0.06, right=0.98,
        hspace=0.2,
    )
    plt.savefig("plots/nonidentifiability.pdf")