"""plot_event options that need no network and no sky map: skymap=False, markers, ax, title, show, legend_kwargs."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import ligo.skymap.plot  # noqa: E402,F401

from GWUtils.models_gw import GWEvent  # noqa: E402
from GWUtils.plot_utils import plot_event  # noqa: E402


def _event():
    return GWEvent(gw_id="GWTEST", group="CBC")


def test_empty_sphere_with_marker():
    fig, ax = _event().plot_event(skymap=False, markers=[dict(ra=30.0, dec=-10.0, label="here")], show=False)
    assert not ax.images and len(ax.lines) == 1
    assert ax.get_title() == "GWTEST" and ax.get_legend() is not None
    plt.close(fig)


def test_title_ax_and_legend_kwargs():
    fig = plt.figure()
    ax = fig.add_subplot(projection="astro hours mollweide")
    markers = [dict(ra=1.0, dec=2.0, label="a", color="red", marker="o")]
    fig2, ax2 = _event().plot_event(ax=ax, skymap=False, markers=markers, title="my title", show=False,
                                    legend_kwargs=dict(loc="upper center"))
    assert ax2 is ax and fig2 is fig and ax.get_title() == "my title"
    assert ax.lines[0].get_marker() == "o"
    plt.close(fig)


def test_no_marker_label_no_legend():
    fig, ax = _event().plot_event(skymap=False, markers=[dict(ra=1.0, dec=2.0)], show=False)
    assert ax.get_legend() is None
    plt.close(fig)


def test_plot_utils_wrapper_passes_keywords():
    fig, ax = plot_event(_event(), skymap=False, markers=[dict(ra=1.0, dec=2.0, label="m")], show=False)
    assert len(ax.lines) == 1
    plt.close(fig)
