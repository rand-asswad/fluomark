from os import makedirs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, PathCollection
from matplotlib.ticker import MaxNLocator


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


def add_data_inset(ax, t_zoom, position, visible, ylim=None, margin=0.12, noticks=True):
    zoom = ax.inset_axes(position, zorder=100)
    zoom.set_facecolor("white")
    y_visible = []
    t0, t1 = t_zoom

    # Copy plotted trajectories
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.shape != y.shape:
            continue

        zoom.plot(x, y, color=line.get_color(), ls=line.get_linestyle(),
                  lw=line.get_linewidth(), alpha=line.get_alpha(),
                  marker=line.get_marker(), zorder=line.get_zorder())

        mask = (x >= t0) & (x <= t1) & np.isfinite(y)
        y_visible.extend(y[mask])

    # Copy shaded intervals and measurement scatter
    for artist in ax.collections:
        if isinstance(artist, PolyCollection):
            vertices = [path.vertices for path in artist.get_paths()]
            zoom.add_collection(PolyCollection(
                vertices, facecolors=artist.get_facecolors(),
                edgecolors=artist.get_edgecolors(),
                linewidths=artist.get_linewidths(),
                alpha=artist.get_alpha(), zorder=artist.get_zorder()))

            for xy in vertices:
                mask = (xy[:, 0] >= t0) & (xy[:, 0] <= t1)
                y_visible.extend(xy[mask, 1])

        elif isinstance(artist, PathCollection):
            xy = np.asarray(artist.get_offsets(), dtype=float)
            zoom.scatter(xy[:, 0], xy[:, 1], s=artist.get_sizes(),
                         facecolors=artist.get_facecolors(),
                         edgecolors=artist.get_edgecolors(),
                         alpha=artist.get_alpha(), zorder=artist.get_zorder())

            mask = (xy[:, 0] >= t0) & (xy[:, 0] <= t1)
            y_visible.extend(xy[mask, 1])

    # Determine vertical limits from data within the time window
    zoom.set_xlim(*t_zoom)
    if ylim is None:
        y_visible = np.asarray(y_visible)
        y_visible = y_visible[np.isfinite(y_visible)]
        low, high = np.min(y_visible), np.max(y_visible)
        span = high - low
        delta = margin * span if span > 0 else 0.05 * max(abs(low), 1e-3)
        ylim = (low - delta, high + delta)

    zoom.set_ylim(*ylim)
    zoom.set_xticks([])
    zoom.yaxis.set_major_locator(MaxNLocator(3))
    zoom.tick_params(axis="y", labelsize=7, pad=1)
    zoom.grid(alpha=0.25, lw=0.5)

    if noticks:
        zoom.set_xticks([])
        zoom.set_yticks([])
        #zoom.tick_params(labelbottom=False, labelleft=False)

    indicator = ax.indicate_inset_zoom(zoom, edgecolor="black")
    connectors = (indicator.connectors if hasattr(indicator, "connectors")
                  else indicator[1])
    for i, connector in enumerate(connectors):
        connector.set_visible(i in visible)
        connector.set_ls(":")
        connector.set_linewidth(0.8)
        connector.set_color("black")

    return zoom