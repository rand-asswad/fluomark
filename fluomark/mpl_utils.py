from os import makedirs
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_mpl_global(output_dir=None):
    width = 7
    ft_size = 14
    mpl.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{bm}\usepackage{amsmath}',
        'mathtext.default': 'regular',
        "font.family": "serif",
        "mathtext.fontset": "cm",
        #"figure.figsize": [width, 0.6 * width],
        "font.size": ft_size,
        "pdf.fonttype": 42,
        #"axes.prop_cycle": mpl.cycler('color', c),
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
        "axes.linewidth": 0.5,
        "axes.formatter.use_mathtext": True,
        "lines.linewidth": 1.5,
        "savefig.dpi": 600,
    })
    if output_dir is not None:
        makedirs(output_dir, exist_ok=True)


LEGEND_KW = dict(fancybox=False, edgecolor="black", facecolor="white", framealpha=1)

def legend(*axs, **kwargs):
    kwargs = {**LEGEND_KW, **kwargs}
    if len(axs) == 0:
        leg = plt.gca().legend(**kwargs)
    elif len(axs) == 1:
        leg = axs[0].legend(**kwargs)
    else:
        all_lines = []
        all_labels = []
        for ax in axs:
            lines, labels = ax.get_legend_handles_labels()
            all_lines += lines
            all_labels += labels
        leg = axs[0].legend(all_lines, all_labels, **kwargs)
    leg.get_frame().set_linewidth(0.5)
    return leg
