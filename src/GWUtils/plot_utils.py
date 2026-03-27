import matplotlib.pyplot as plt
import ligo.skymap.plot
from pathlib import Path
from GWUtils.models_gw import GWEvent
import matplotlib.patches as patches
import numpy as np
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.coordinates import SkyCoord
from astropy import units as u

def plot_event(event: GWEvent, figPath: Path | str | None = None, circle_roi: bool = False, rect_roi : bool = False):
    skymap, meta = event.load_skymap()
    fig = plt.figure(figsize=(9, 4), dpi=100)
    ax = plt.axes(projection='astro hours mollweide')
    ax.grid()
    ax.imshow_hpx(skymap, cmap='cylon')

    if circle_roi:
        roi = event.get_90_roi_circle()
        circle = SphericalCircle(
            center=SkyCoord(roi['ra']* u.deg, roi['dec'] * u.deg),
            radius=roi['radius_deg'] * u.deg,
            transform=ax.get_transform('icrs'),
            edgecolor='white', facecolor='none',
            linewidth=1.5, linestyle='--', label='90% CI'
        )
        ax.add_patch(circle)
        ax.legend(loc='lower right')
    if rect_roi:
        roi = event.get_90_roi_rect()
        ra_min, ra_max = roi['ra_min'], roi['ra_max']
        dec_min, dec_max = roi['dec_min'], roi['dec_max']

        # Draw the 4 edges as lines in icrs coordinates to handle projection correctly
        ra_top    = np.linspace(ra_min, ra_max, 100)
        ra_bottom = np.linspace(ra_min, ra_max, 100)
        dec_left  = np.linspace(dec_min, dec_max, 100)
        dec_right = np.linspace(dec_min, dec_max, 100)

        transform = ax.get_transform('icrs')
        kwargs = dict(transform=transform, color='cyan', linewidth=1.5, linestyle='--')

        ax.plot(ra_top,             np.full(100, dec_max), **kwargs)  # top
        ax.plot(ra_bottom,          np.full(100, dec_min), **kwargs)  # bottom
        ax.plot(np.full(100, ra_min), dec_left,            **kwargs)  # left
        ax.plot(np.full(100, ra_max), dec_right,           **kwargs, label='90% bbox')  # right

        ax.legend(loc='lower right')
    

    for a in [ax]:
        a.set_facecolor('white')
        for key in ['ra', 'dec']:
            a.coords[key].set_auto_axislabel(False)

    ax.set_title(event.superevent_id)

    if figPath:
        fig.savefig(figPath, dpi=300)
    plt.show()
    return fig, ax

def plot_events(events: list[GWEvent], figPath: Path | str | None = None):
    fig = plt.figure(figsize=(9, 4), dpi=100)
    ax = plt.axes(projection='astro hours mollweide')
    ax.grid()
    ax.set_facecolor('white')

    cmaps = ['cylon', 'Blues', 'Greens', 'Oranges', 'Purples']  # one per event

    for event, cmap in zip(events, cmaps):
        skymap, meta = event.load_skymap()
        ax.imshow_hpx(skymap, cmap=cmap, alpha=0.5)

    for key in ['ra', 'dec']:
        ax.coords[key].set_auto_axislabel(False)

    ax.legend(handles=[
        patches.Patch(label=e.superevent_id, color=plt.get_cmap(c)(0.6))
        for e, c in zip(events, cmaps)
    ])

    plt.tight_layout()
    if figPath:
        fig.savefig(figPath, dpi=300)
    plt.show()
    return fig, ax