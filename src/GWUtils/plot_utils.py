import matplotlib.pyplot as plt
import ligo.skymap.plot
from pathlib import Path
from GWUtils.models_gw import GWEvent
import matplotlib.patches as patches
import numpy as np
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.coordinates import SkyCoord
from astropy import units as u


def plot_event(
    event: GWEvent,
    figPath: Path | str | None = None,
    circle_roi: bool = False,
    rect_roi: bool = False,
    n_vertices: int | None = None,
    **kwargs,
):
    """Plot the skymap of `event`, with optional 90% credible-region overlays.

    Thin wrapper around `GWEvent.plot_event` (which takes the extra keywords
    `ax`, `skymap`, `markers`, `title`, `show`, passed through `kwargs`).
    """
    return event.plot_event(
        figPath=figPath,
        circle_roi=circle_roi,
        rect_roi=rect_roi,
        n_vertices=n_vertices,
        **kwargs,
    )


def plot_events(events: list[GWEvent], figPath: Path | str | None = None):
    fig = plt.figure(figsize=(9, 4), dpi=100)
    ax = plt.axes(projection="astro hours mollweide")
    ax.grid()
    ax.set_facecolor("white")

    cmaps = ["cylon", "Blues", "Greens", "Oranges", "Purples"]  # one per event

    for event, cmap in zip(events, cmaps):
        skymap, meta = event.load_skymap()
        ax.imshow_hpx(skymap, cmap=cmap, alpha=0.5)

    for key in ["ra", "dec"]:
        ax.coords[key].set_auto_axislabel(False)

    ax.legend(
        handles=[
            patches.Patch(label=e.superevent_id, color=plt.get_cmap(c)(0.6))
            for e, c in zip(events, cmaps)
        ]
    )

    plt.tight_layout()
    if figPath:
        fig.savefig(figPath, dpi=300)
    plt.show()
    return fig, ax
