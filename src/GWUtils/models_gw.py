from pydantic import BaseModel, Field, field_validator, ConfigDict, PrivateAttr
from typing import List, Optional, ClassVar, Any
from pathlib import Path
import datetime
import json
import re
import socket
from enum import Enum
from astropy.time import Time
from urllib.request import urlretrieve
from .define import SKYMAP_FITS_DIRECTORY, EVENTS_DIRECTORY
from ligo.gracedb.rest import GraceDb, HTTPError
from ligo.skymap.io import fits
from ligo.skymap.postprocess import find_greedy_credible_levels
import astropy_healpix as ah
import numpy as np
from astropy import units as u
import healpy as hp
from ligo.skymap.io import read_sky_map
from gwosc.api import fetch_event_json
import gwosc.datasets as datasets
import matplotlib.pyplot as plt
import ligo.skymap.plot
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.coordinates import SkyCoord
import pandas as pd
from mocpy import MOC
from contextlib import contextmanager

# GraceDB/GWOSC calls below don't set a socket timeout themselves, so on a
# filtered/dead network they hang indefinitely instead of raising a fast
# ConnectionError. Bound them so the offline-cache fallback actually kicks in.
NETWORK_TIMEOUT_SECONDS = 10


@contextmanager
def _bounded_network_timeout(seconds: float = NETWORK_TIMEOUT_SECONDS):
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


class Detector(Enum):
    LIGO_Hanford = "H1"
    LIGO_Livingston = "L1"
    Virgo = "V1"
    KAGRA = "K1"

    def __str__(self):
        return self.name


class UncertainQuantity(BaseModel):
    """A measured value with asymmetric uncertainties."""

    value: float
    lower: Optional[float] = None  # negative convention (e.g. -1.3)
    upper: Optional[float] = None
    unit: Optional[str] = None


def _uq(val, lower, upper, unit) -> Optional[UncertainQuantity]:
    if val is None:
        return None
    return UncertainQuantity(value=val, lower=lower, upper=upper, unit=unit)


def _uq_to_dict(field_name: str, uq: Optional[UncertainQuantity]) -> dict:
    """Flatten an UncertainQuantity into prefixed dict keys."""
    if uq is None:
        return {
            field_name: None,
            f"{field_name}_lower": None,
            f"{field_name}_upper": None,
        }
    return {
        field_name: uq.value,
        f"{field_name}_lower": uq.lower,
        f"{field_name}_upper": uq.upper,
    }


class CBCClassification(BaseModel):
    astro: Optional[float] = None
    terrestrial: Optional[float] = Field(None, alias="Terrestrial")
    bbh: Optional[float] = Field(None, alias="BBH")
    bns: Optional[float] = Field(None, alias="BNS")
    nsbh: Optional[float] = Field(None, alias="NSBH")
    source_pipeline: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)

    def most_probable(self) -> tuple[str, float]:
        """Return (classification_name, probability) for the highest probability class."""
        candidates = {
            "BBH": self.bbh,
            "BNS": self.bns,
            "NSBH": self.nsbh,
            "Terrestrial": self.terrestrial,
        }
        # Filter out None values
        valid = {k: v for k, v in candidates.items() if v is not None}
        if not valid:
            return ("Unknown", 0.0)
        best = max(valid, key=valid.get)
        return (best, valid[best])

    def is_astrophysical(self, threshold: float = 0.5) -> bool:
        """True if total astrophysical probability exceeds threshold."""
        return (self.astro or 0.0) >= threshold


class GWEvent(BaseModel):
    """
    GWEvent model for parsing LIGO/Virgo/KAGRA superevent data.

    The sky map is taken to be the BAYESTARS one by default.
    Another method than simple model validation is needed to extract
    the various available skymaps from Gracedb, and update the model
    with the most precise skymap to date (e.g. from LALInference or Bilby) when available.

    Indeed for CBCs it seems we can query the bayestar.multiorder.fits skymap directly,
    but for other types of events (e.g. burst) the bayestar skymap is not available,
    but probably the cwb.multiorder.fits skymap is, and should be used instead.
    """

    # Identity 
    superevent_id: Optional[str] = None
    gw_id: Optional[str] = None
    catalog: Optional[str] = None

    # Timing 
    created: Optional[datetime.datetime] = None
    t_start: Optional[datetime.datetime] = None
    t_end: Optional[datetime.datetime] = None
    t_0: Optional[datetime.datetime] = None
    gps: Optional[float] = None

    # Detectors & pipeline
    detectors: List[Detector] = []
    group: Optional[str] = None
    preferred_event: Optional[str] = None
    network_snr: Optional[float] = None

    # Rates
    far: Optional[float] = None
    p_astro: Optional[float] = None

    # Status flags 
    skymap_ready: bool = False
    pastro_ready: bool = False

    # Classification
    classification: Optional[CBCClassification] = None

    # Masses [M_sun] 
    mass_1: Optional[UncertainQuantity] = None
    mass_2: Optional[UncertainQuantity] = None
    chirp_mass: Optional[UncertainQuantity] = None
    total_mass: Optional[UncertainQuantity] = None
    final_mass: Optional[UncertainQuantity] = None

    # Distance & cosmology
    luminosity_distance: Optional[UncertainQuantity] = None  # [Mpc]
    redshift: Optional[UncertainQuantity] = None

    # Spins 
    chi_eff: Optional[UncertainQuantity] = None

    # metadata
    preferred_waveform: Optional[str] = None
    posterior_url: Optional[str] = None
    strain_files: Optional[list[dict]] = None

    # Skymap 
    skymap_path: Optional[str | Path] = None
    _skymap: Optional[Any] = PrivateAttr(default=None)
    _meta: Optional[Any] = PrivateAttr(default=None)

    model_config: ClassVar = {"extra": "ignore"}

    # Suffixes of actual GraceDB filenames, in order of preference. Matched
    # with `str.endswith`, so e.g. "PublicationSamples.multiorder.fits" also
    # matches event-specific names like "GW190503_185404_PublicationSamples.multiorder.fits".
    # Older (O1-O3a) events often only have a flat bayestar.fits(.gz), no
    # Bilby/LALInference multiorder file, hence the long fallback chain.
    skymap_priorities_map: ClassVar = {
        "CBC": [
            "Bilby.multiorder.fits",
            "LALInference.multiorder.fits",
            "LALInference.offline.multiorder.fits",
            "PublicationSamples.multiorder.fits",
            "bayestar.multiorder.fits",
            "bayestar.fits.gz",
            "bayestar.fits",
        ],
        "Burst": [
            "cwb.multiorder.fits",
            "cwb.LHV.multiorder.fits",
            "cwb.fits.gz",
            "cwb.fits",
        ],
    }

    # O1/O2 events predate GraceDB's skymap distribution; LVC released their
    # skymaps separately as LIGO-P1800381 (https://dcc.ligo.org/LIGO-P1800381).
    gwtc1_skymap_url: ClassVar = (
        "https://dcc.ligo.org/public/0157/P1800381/007/{name}_skymap.fits.gz"
    )

    def __new__(cls, *args, **kwargs):
        """Allow `GWEvent(identifier)` as a shorthand fetch: a bare string is
        resolved to a GWTCEvent (GWTC catalog name, e.g. 'GW170817') or a
        GWEvent (GraceDB superevent ID, e.g. 'S250328ae'). Regular field-kwarg
        construction and model_validate() are unaffected.

        Pass `offline=True` to skip GraceDB/GWOSC entirely and load the event
        from the local JSON cache written by `.save()` instead. Even with the
        default `offline=False`, a network failure (no connection, DNS, etc.)
        transparently falls back to that same cache when available."""
        if len(args) == 1 and isinstance(args[0], str):
            return _build_from_identifier(args[0], **kwargs)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            return  # already fully built in __new__
        super().__init__(*args, **kwargs)

    @field_validator("detectors", mode="before")
    def validate_detectors(cls, v):
        if isinstance(v, list):
            return [Detector(det) for det in v]
        elif isinstance(v, str):
            return [Detector(det) for det in v.split(",")]
        raise ValueError(
            "Detectors must be a list of strings or a string of comma-separated values"
        )

    @field_validator("created", mode="before")
    @classmethod
    def parse_created(cls, v):
        if isinstance(v, str):
            if ("T" in v) & ("UTC" not in v):
                return datetime.datetime.fromisoformat(v.replace("Z", "+00:00"))
            try:
                dt_str = v.replace(" UTC", "").strip()
                dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                return dt.replace(tzinfo=datetime.timezone.utc)
            except Exception as e:
                raise e
        return v

    @field_validator("t_start", "t_end", "t_0", mode="before")
    @classmethod
    def parse_gps_time(cls, v):
        """Convert GPS time to datetime."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            t = Time(v, format="gps", scale="utc")
            return t.to_datetime(timezone=datetime.timezone.utc)
        return v

    @classmethod
    def model_validate(cls, obj, **kwargs):
        """Extract nested fields and process labels."""
        obj = dict(obj)  # Make a copy to modify

        # If already a flat model dump, skip GraceDB-specific extraction
        if "preferred_event_data" not in obj:
            return super().model_validate(obj, **kwargs)

        # Extract labels
        labels = obj.get("labels", [])
        obj["labels"] = labels

        # Extract instruments from preferred_event_data
        preferred = obj.get("preferred_event_data", {})
        if preferred:
            obj["detectors"] = preferred.get("instruments")
            obj["group"] = preferred.get("group")
            obj["preferred_event"] = preferred.get("graceid")

        obj["skymap_ready"] = (
            "SKYMAP_READY" in labels or "SKYMAP_READY" in preferred.get("labels", [])
        )
        obj["pastro_ready"] = (
            "PASTRO_READY" in labels or "PASTRO_READY" in preferred.get("labels", [])
        )

        # If skymap downloaded, fetch the filename
        skymap_path = Path(SKYMAP_FITS_DIRECTORY) / f"{obj['superevent_id']}*.fits"
        matching_files = list(
            Path(SKYMAP_FITS_DIRECTORY).glob(f"{obj['superevent_id']}_*.fits")
        )
        obj["skymap_path"] = str(matching_files[0]) if matching_files else None

        return super().model_validate(obj, **kwargs)

    @classmethod
    def from_gwosc(cls, gw_name: str) -> "GWEvent":
        """Construct directly from a GWOSC catalog name, no GraceDB needed."""
        # Lazy import to avoid circular import
        from .query_utils import query_latest_gwtc_dataset

        if '-v' not in gw_name :     # Avoid case where version is already indicated
            res = query_latest_gwtc_dataset(gw_name)
            if len(res) != 1:
                raise ValueError(f"Expected one answer, had {res} of length {len(res)}")

            gw_name = res[0]

        ev = fetch_event_json(gw_name)
        events = ev.get("events", {})
        if not events:
            raise ValueError(f"No GWOSC data for {gw_name}")

        latest = max(events.values(), key=lambda e: e.get("version", 0))
        preferred_pe = next(
            (p for p in latest["parameters"].values() if p.get("is_preferred")), None
        )

        return GWEvent(
            superevent_id=latest.get("gracedb_id") or gw_name,
            gw_id=gw_name,
            catalog=latest.get("catalog.shortName"),
            gps=latest.get("GPS"),
            t_0=latest.get("GPS"),
            far=latest.get("far"),
            p_astro=latest.get("p_astro"),
            network_snr=latest.get("network_matched_filter_snr"),
            mass_1=_uq(
                latest.get("mass_1_source"),
                latest.get("mass_1_source_lower"),
                latest.get("mass_1_source_upper"),
                latest.get("mass_1_source_unit"),
            ),
            mass_2=_uq(
                latest.get("mass_2_source"),
                latest.get("mass_2_source_lower"),
                latest.get("mass_2_source_upper"),
                latest.get("mass_2_source_unit"),
            ),
            chirp_mass=_uq(
                latest.get("chirp_mass_source"),
                latest.get("chirp_mass_source_lower"),
                latest.get("chirp_mass_source_upper"),
                latest.get("chirp_mass_source_unit"),
            ),
            total_mass=_uq(
                latest.get("total_mass_source"),
                latest.get("total_mass_source_lower"),
                latest.get("total_mass_source_upper"),
                latest.get("total_mass_source_unit"),
            ),
            final_mass=_uq(
                latest.get("final_mass_source"),
                latest.get("final_mass_source_lower"),
                latest.get("final_mass_source_upper"),
                latest.get("final_mass_source_unit"),
            ),
            luminosity_distance=_uq(
                latest.get("luminosity_distance"),
                latest.get("luminosity_distance_lower"),
                latest.get("luminosity_distance_upper"),
                latest.get("luminosity_distance_unit"),
            ),
            redshift=_uq(
                latest.get("redshift"),
                latest.get("redshift_lower"),
                latest.get("redshift_upper"),
                latest.get("redshift_unit"),
            ),
            chi_eff=_uq(
                latest.get("chi_eff"),
                latest.get("chi_eff_lower"),
                latest.get("chi_eff_upper"),
                latest.get("chi_eff_unit"),
            ),
            preferred_waveform=(
                preferred_pe.get("waveform_family") if preferred_pe else None
            ),
            posterior_url=preferred_pe.get("data_url") if preferred_pe else None,
            strain_files=latest.get("strain"),
            skymap_ready=False,
            pastro_ready=False,
        )

    def _resolve_gwtc_name(self) -> Optional[str]:
        """Find the GWTC catalog name matching this event's coalescence time.

        GWOSC has no lookup by GraceDB superevent ID, so for events built
        from a bare superevent ID (no gw_id known yet) we match on GPS time
        instead, via `find_datasets`'s `segment` filter.
        """
        if self.t_0 is None:
            return None
        gps = Time(self.t_0).gps
        candidates = datasets.find_datasets(type="event", segment=(gps - 1, gps + 1))
        if not candidates:
            return None
        candidates.sort(key=lambda c: int(c.rsplit("-v", 1)[1]) if "-v" in c else 0)
        return candidates[-1]

    def enrich_from_gwosc(self) -> "GWEvent":
        """
        Overlay catalog-quality GWOSC parameters onto a GraceDB-sourced event.
        Preserves GraceDB identity, timing, skymap status, and classification.
        """
        name = self.gw_id or self._resolve_gwtc_name()
        if name is None:
            print(f"debug : no GWOSC catalog match found for {self.superevent_id}")
            return self
        try:
            enriched = GWEvent.from_gwosc(name)
            pe_fields = [
                "catalog",
                "gps",
                "far",
                "p_astro",
                "network_snr",
                "mass_1",
                "mass_2",
                "chirp_mass",
                "total_mass",
                "final_mass",
                "luminosity_distance",
                "redshift",
                "chi_eff",
                "preferred_waveform",
                "posterior_url",
                "strain_files",
            ]
            for field in pe_fields:
                val = getattr(enriched, field)
                if val is not None:
                    setattr(self, field, val)
        except Exception as e:
            print(f"debug : exception when enriching from gwotc : {e}")
        return self

    def has_dl_skymap(self) -> bool | Path:
        """Check if the skymap has already been downloaded locally."""
        if self.skymap_path is not None and Path(self.skymap_path).exists():
            return Path(self.skymap_path)
        matching = sorted(Path(SKYMAP_FITS_DIRECTORY).glob(f"{self.identifier}*.fits*"))
        if matching:
            return matching[0]
        return False

    def _is_gwtc1_event(self) -> bool:
        """O1/O2 (GWTC-1) events have no GraceDB superevent or skymap."""
        return bool(self.gw_id) and bool(self.catalog) and self.catalog.startswith("GWTC-1")

    def _download_gwtc1_skymap(self):
        """Fetch the skymap from the public GWTC-1 data release (LIGO-P1800381),
        since these events were never assigned a GraceDB superevent/skymap."""
        remote_name = re.sub(r"-v\d+$", "", self.gw_id)
        url = self.gwtc1_skymap_url.format(name=remote_name)
        filename = Path(SKYMAP_FITS_DIRECTORY) / f"{self.identifier}_skymap.fits.gz"
        try:
            urlretrieve(url, filename)
            print(f"Skymap downloaded successfully from: {url}")
            self.skymap_path = filename
            self.save()
            return filename
        except Exception as e:
            print(f"Failed to download GWTC-1 skymap from: {url}. Error: {e}")

    def _best_skymap_filename(self, available: list[str]) -> Optional[str]:
        """Pick the best skymap filename actually present, by group priority."""
        if self.group is None:
            return None
        patterns = GWEvent.skymap_priorities_map.get(self.group)
        if patterns is None:
            return None
        for pattern in patterns:
            for filename in available:
                if filename.endswith(pattern):
                    return filename
        return None

    def download_skymap(self):

        if self._is_gwtc1_event():
            return self._download_gwtc1_skymap()

        if not self.skymap_ready:
            print("Skymap is not ready. Cannot download skymap.")
            return
        if self.group is None:
            print("Group information is missing. Cannot determine skymap URL.")
            return

        client = GraceDb()
        try:
            available = list(client.files(self.superevent_id).json().keys())
        except Exception as e:
            print(f"Failed to list files for {self.superevent_id}: {e}")
            return

        filename = self._best_skymap_filename(available)
        if filename is None:
            print(
                f"No matching skymap file for group {self.group} among: {available}"
            )
            return

        # Use the authenticated client (not a bare urlretrieve) since most
        # GraceDB file downloads require LIGO.ORG credentials.
        local_path = Path(SKYMAP_FITS_DIRECTORY) / f"{self.superevent_id}_{filename}"
        try:
            with open(local_path, "wb") as f:
                f.write(client.files(self.superevent_id, filename).read())
            print(f"Skymap downloaded successfully: {filename}")
            self.skymap_path = local_path
            self.save()
            return local_path
        except Exception as e:
            print(f"Failed to download skymap file {filename}: {e}")

    def load_skymap(self):
        if self._skymap is None:

            skymap_path = self.has_dl_skymap()

            if not skymap_path:
                skymap_path = self.download_skymap()

            print(f"DEBUG : path = {skymap_path}")
            skymap, meta = read_sky_map(skymap_path)
            self._skymap = skymap
            self._meta = meta

        return self._skymap, self._meta

    def unload_skymap(self):

        if self._skymap is not None:
            del self._skymap
        if self._meta is not None:
            del self._meta

        self._skymap, self._meta = None, None

    def get_90_roi_rect(self):
        """Returns (ra_min, ra_max, dec_min, dec_max) of the 90% credible region."""
        skymap, meta = self.load_skymap()

        # Find credible levels for each pixel
        credible_levels = find_greedy_credible_levels(skymap)

        # Get pixels inside 90% credible region
        nside = ah.npix_to_nside(len(skymap))
        inside_90 = np.where(credible_levels <= 0.9)[0]

        # Convert pixel indices to RA/Dec
        ra, dec = ah.healpix_to_lonlat(inside_90, nside)
        ra = ra.to(u.deg).value
        dec = dec.to(u.deg).value

        return {
            "superevent_id": self.superevent_id,
            "ra_min": ra.min(),
            "ra_max": ra.max(),
            "dec_min": dec.min(),
            "dec_max": dec.max(),
            "area_deg2": len(inside_90)
            * ah.nside_to_pixel_area(nside).to(u.deg**2).value,
        }

    def get_90_roi_circle(self):
        """Returns the smallest circle (ra, dec, radius_deg) containing 90% credible region."""
        skymap, meta = self.load_skymap()
        nside = hp.npix2nside(len(skymap))

        credible_levels = find_greedy_credible_levels(skymap)

        # Center on the maximum probability pixel
        best_pixel = np.argmax(skymap)
        theta, phi = hp.pix2ang(nside, best_pixel)

        # Convert to ra/dec
        ra_center = np.rad2deg(phi)
        dec_center = 90 - np.rad2deg(theta)
        xyz_center = hp.ang2vec(theta, phi)

        # Binary search for the smallest radius that captures 90% probability
        radius_min, radius_max = 0.0, np.pi  # radians
        for _ in range(50):  # 50 iterations is more than enough precision
            radius_mid = 0.5 * (radius_min + radius_max)
            ipix = hp.query_disc(nside, xyz_center, radius_mid)
            prob = skymap[ipix].sum()
            if prob < 0.9:
                radius_min = radius_mid
            else:
                radius_max = radius_mid

        return {
            "superevent_id": self.superevent_id,
            "ra": ra_center,
            "dec": dec_center,
            "radius_deg": np.rad2deg(radius_max),
        }

    def _credible_moc(self, percentile: float, n_vertices: int) -> tuple[MOC, int]:
        """Build the exact credible-region MOC, then coarsen it just enough
        for its boundary to have at most `n_vertices` vertices in total.

        MOC coarsening only ever grows a region (a coarse cell is kept whole
        as soon as any of its finer sub-cells belongs to the MOC), so the
        returned region is always a superset of the true credible area,
        never an underestimate, no matter how small `n_vertices` is.
        """
        skymap, _ = self.load_skymap()
        nside = ah.npix_to_nside(len(skymap))
        level = ah.nside_to_level(nside)

        ipix_nest = hp.ring2nest(nside, np.arange(len(skymap)))
        uniq = ah.level_ipix_to_uniq(level, ipix_nest)

        moc = MOC.from_valued_healpix_cells(
            uniq.astype(np.uint64),
            skymap.astype(np.float64),
            max_depth=level,
            cumul_to=percentile / 100.0,
        )

        for order in range(level, -1, -1):
            n_boundary = sum(len(c) for c in moc.get_boundaries(order=order))
            if n_boundary <= n_vertices or order == 0:
                return moc.degrade_to_order(order), order

    def get_roi(self, percentile: float = 90, n_vertices: int = 20, format: str = "dict"):
        """Region of interest (credible region) for this event.

        Built as an exact MOC (mocpy) from the skymap, then coarsened so its
        boundary has at most `n_vertices` points — coarsening always grows
        the region rather than shrinking it (see `_credible_moc`), so the
        returned area is never an underestimate of the true `percentile`%
        credible region.

        Parameters
        ----------
        percentile : float
            Credible level in percent (e.g. 90 for the 90% credible region).
        n_vertices : int
            Target maximum number of boundary vertices, summed across all
            disjoint contours.
        format : {'dict', 'moc'}
            'dict' returns a JSON-friendly summary with the boundary contours
            (ra/dec in degrees); 'moc' returns the `mocpy.MOC` object itself.
        """
        if format not in ("dict", "moc"):
            raise ValueError(f"format must be 'dict' or 'moc', got {format!r}")

        moc, order = self._credible_moc(percentile, n_vertices)
        if format == "moc":
            return moc

        contours = [
            list(zip(c.ra.deg.tolist(), c.dec.deg.tolist()))
            for c in moc.get_boundaries(order=order)
        ]
        return {
            "superevent_id": self.superevent_id,
            "percentile": percentile,
            "order": order,
            "area_deg2": moc.sky_fraction * 4 * 180**2 / np.pi,
            "n_vertices": sum(len(c) for c in contours),
            "contours": contours,
        }

    def get_type_CBC(self):
        """Returns most probable type of CBC event, relying on classification"""
        if (not self.group == "CBC") | (self.classification is None):
            return "Unknown"

        elif self.classification is not None:
            label, prob = self.classification.most_probable()
            return label, prob

    @property
    def identifier(self) -> Optional[str]:
        """Best available tag for this event: the GraceDB superevent ID, falling
        back to the GWTC catalog name for catalog-only (pre-superevent) events."""
        return self.superevent_id or self.gw_id

    def save(self):
        Path(EVENTS_DIRECTORY).mkdir(parents=True, exist_ok=True)
        with open(Path(EVENTS_DIRECTORY) / f"{self.identifier}.json", "w") as file:
            file.write(self.model_dump_json())

    def plot_event(
        self,
        figPath: Path | str | None = None,
        circle_roi: bool = False,
        rect_roi: bool = False,
        n_vertices: Optional[int] = None,
    ):
        """Plot the skymap, with optional 90% credible-region overlays.

        `n_vertices`, if given, overlays the MOC-based ROI from `get_roi`
        (see its docstring) coarsened to that many boundary vertices,
        drawn as one closed polygon per disjoint credible-region contour.
        """
        skymap, meta = self.load_skymap()
        fig = plt.figure(figsize=(9, 4), dpi=100)
        ax = plt.axes(projection="astro hours mollweide")
        ax.grid()
        ax.imshow_hpx(skymap, cmap="cylon")
        transform = ax.get_transform("icrs")
        has_legend = False

        if circle_roi:
            roi = self.get_90_roi_circle()
            circle = SphericalCircle(
                center=SkyCoord(roi["ra"] * u.deg, roi["dec"] * u.deg),
                radius=roi["radius_deg"] * u.deg,
                transform=transform,
                edgecolor="white",
                facecolor="none",
                linewidth=1.5,
                linestyle="--",
                label="90% CI",
            )
            ax.add_patch(circle)
            has_legend = True
        if rect_roi:
            roi = self.get_90_roi_rect()
            ra_min, ra_max = roi["ra_min"], roi["ra_max"]
            dec_min, dec_max = roi["dec_min"], roi["dec_max"]

            # Draw the 4 edges as lines in icrs coordinates to handle projection correctly
            ra_top = np.linspace(ra_min, ra_max, 100)
            ra_bottom = np.linspace(ra_min, ra_max, 100)
            dec_left = np.linspace(dec_min, dec_max, 100)
            dec_right = np.linspace(dec_min, dec_max, 100)

            kwargs = dict(
                transform=transform, color="cyan", linewidth=1.5, linestyle="--"
            )

            ax.plot(ra_top, np.full(100, dec_max), **kwargs)  # top
            ax.plot(ra_bottom, np.full(100, dec_min), **kwargs)  # bottom
            ax.plot(np.full(100, ra_min), dec_left, **kwargs)  # left
            ax.plot(
                np.full(100, ra_max), dec_right, **kwargs, label="90% bbox"
            )  # right
            has_legend = True
        if n_vertices is not None:
            roi = self.get_roi(percentile=90, n_vertices=n_vertices, format="dict")
            kwargs = dict(transform=transform, color="lime", linewidth=1.5, linestyle="--")
            for i, contour in enumerate(roi["contours"]):
                ra, dec = zip(*contour)
                ra, dec = ra + (ra[0],), dec + (dec[0],)  # close the polygon
                label = f"{roi['percentile']:.0f}% MOC" if i == 0 else None
                ax.plot(ra, dec, label=label, **kwargs)
            has_legend = True

        if has_legend:
            ax.legend(loc="lower right")

        for a in [ax]:
            a.set_facecolor("white")
            for key in ["ra", "dec"]:
                a.coords[key].set_auto_axislabel(False)

        name = self.gw_id if self.gw_id else self.superevent_id
        ax.set_title(name)

        if figPath:
            fig.savefig(figPath, dpi=300)
        plt.show()
        return fig, ax

    def to_dict(self) -> dict:
        """Flatten the event to a plain dict, expanding UncertainQuantity fields."""
        uq_fields = {
            "mass_1",
            "mass_2",
            "chirp_mass",
            "total_mass",
            "final_mass",
            "luminosity_distance",
            "redshift",
            "chi_eff",
        }
        d = {}
        for field_name, value in self:
            if field_name in uq_fields:
                d.update(_uq_to_dict(field_name, value))
            elif field_name == "classification" and value is not None:
                d["bbh"] = value.bbh
                d["bns"] = value.bns
                d["nsbh"] = value.nsbh
                d["terrestrial"] = value.terrestrial
            elif field_name == "detectors":
                d["detectors"] = ",".join(det.value for det in value)
            elif field_name == "strain_files":
                pass  # too nested to be useful in a flat dataframe
            else:
                d[field_name] = value
        return d

    def to_dataframe(self) -> pd.DataFrame:
        """Return a single-row DataFrame for this event."""
        return pd.DataFrame([self.to_dict()])


class GWTCEvent(GWEvent):
    """
    A GWEvent instantiated directly from a GWTC catalog name.
    Merges GWOSC catalog parameters with GraceDB superevent data.
    Usage:
        ev = GWTCEvent("GW230627_015337")
        evs = GWTCEvent(["GW230627_015337", "GW230919_215712"])
    """

    def __new__(
        cls,
        gw_name: str | list[str],
        client=None,
        classification: bool = True,
        offline: bool = False,
    ):
        if isinstance(gw_name, list):
            return [
                _build_gwevent_from_gw_name(
                    name, cls=cls, client=client, classification=classification, offline=offline
                )
                for name in gw_name
            ]
        return _build_gwevent_from_gw_name(
            gw_name, cls=cls, client=client, classification=classification, offline=offline
        )

    def __init__(self, *args, **kwargs):
        """No-op: the instance is already fully built and validated in __new__."""


if __name__ == "__main__":

    example = {
        "superevent_id": "S251117dq",
        "gw_id": None,
        "category": "Production",
        "created": "2025-11-17 21:38:45 UTC",
        "t_start": 1447450730.097656,
        "t_0": 1447450731.119385,
        "t_end": 1447450732.121005,
        "far": 5.867636084251777e-15,
        "labels": ["EM_READY", "PE_READY", "SKYMAP_READY"],
        "preferred_event_data": {
            "instruments": "H1,L1",
        },
    }

    # Parse it
    event: GWEvent = GWEvent.model_validate(example)

    # Access the fields
    print(f"Event ID: {event.superevent_id}")
    print(f"Created: {event.created}")
    print(f"Coalescence: {event.t_0}")
    print(f"FAR: {event.far:.2e} Hz")
    print(f"Skymap ready: {event.skymap_ready}")
    print(f"Detectors: {event.detectors}")
    print(f"Duration: {(event.t_end - event.t_start).total_seconds():.3f} s")


def is_classification_json(data: dict) -> bool:
    """Detects if a dict looks like a Classification"""
    keys = set(data.keys())

    expected = {"Astro", "Terrestrial", "BBH", "BNS", "NSBH"}

    return len(keys.intersection(expected)) >= 2


def _fetch_classification(sev: GWEvent, client) -> None:
    """Fetch and attach p_astro classification from GraceDB files. Mutates sev in place."""
    if not sev.pastro_ready:
        return
    try:
        # Use the superevent's file list, not the preferred (G-)event's: the
        # latter requires GraceDB permissions beyond plain superevent read
        # access and 401s for accounts that can still read the superevent
        # fine, whereas the superevent's own file list already includes the
        # p_astro.json produced for the preferred event.
        files_dict = client.files(sev.superevent_id).json()
        for filename in files_dict:
            if filename.endswith(".json") and "p_astro" in filename:
                data = client.files(sev.superevent_id, filename).json()
                if is_classification_json(data):
                    sev.classification = CBCClassification.model_validate(data)
                    break
    except Exception as e:
        print(f"Could not fetch classification for {sev.superevent_id}: {e}")


def _strip_gwtc_version(name: Optional[str]) -> Optional[str]:
    """'GW170817-v3' -> 'GW170817', so lookups by unversioned name still match."""
    return re.sub(r"-v\d+$", "", name) if name else name


def _load_cached_event(identifier: str, events_dir: Path = EVENTS_DIRECTORY) -> Optional[GWEvent]:
    """Look up a previously cached GWEvent (written by GWEvent.save()) by
    superevent_id or GWTC name (version-suffix-insensitive). Returns None if
    no cached record matches."""
    direct = Path(events_dir) / f"{identifier}.json"
    if direct.exists():
        return GWEvent.model_validate(json.loads(direct.read_text()))
    for path in sorted(Path(events_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if identifier in (data.get("superevent_id"), data.get("gw_id")):
            return GWEvent.model_validate(data)
        if _strip_gwtc_version(identifier) == _strip_gwtc_version(data.get("gw_id")):
            return GWEvent.model_validate(data)
    return None


def _build_from_identifier(
    identifier: str, client=None, classification: bool = True, offline: bool = False
) -> GWEvent:
    """Resolve a bare identifier to a fully built event: a GWTC catalog name
    (e.g. 'GW170817') yields a GWTCEvent, anything else is treated as a
    GraceDB superevent ID (e.g. 'S250328ae') and yields a plain GWEvent."""
    if identifier.upper().startswith("GW"):
        return _build_gwevent_from_gw_name(
            identifier, cls=GWTCEvent, client=client, classification=classification, offline=offline
        )
    return _build_gwevent(identifier, client=client, classification=classification, offline=offline)


def _build_gwevent(
    superevent_id: str, client=None, classification: bool = True, offline: bool = False
) -> GWEvent:
    if offline:
        cached = _load_cached_event(superevent_id)
        if cached is None:
            raise RuntimeError(f"No cached event found for {superevent_id!r} (offline mode)")
        return cached
    try:
        with _bounded_network_timeout():
            if client is None:
                client = GraceDb()
            rep = client.superevent(superevent_id)
            sev = GWEvent.model_validate(rep.json())
            if classification:
                _fetch_classification(sev, client)
            sev.enrich_from_gwosc()
        sev.save()
        return sev
    except OSError as e:
        cached = _load_cached_event(superevent_id)
        if cached is None:
            raise
        print(f"Network unavailable ({e}); falling back to cached event for {superevent_id}")
        return cached


def _build_gwevent_from_gw_name(
    gw_name: str,
    cls: type[GWEvent] = GWEvent,
    client=None,
    classification: bool = True,
    offline: bool = False,
) -> GWEvent:
    if offline:
        cached = _load_cached_event(gw_name)
        if cached is None:
            raise RuntimeError(f"No cached event found for {gw_name!r} (offline mode)")
        cached.__class__ = cls
        return cached
    try:
        with _bounded_network_timeout():
            if client is None:
                client = GraceDb()
            gwosc = GWEvent.from_gwosc(gw_name)
            try:
                rep = client.superevent(gwosc.superevent_id)
            except HTTPError:
                rep = client.event(gwosc.superevent_id)
            # Always validate as the base class: cls may override __new__/__init__
            # (see GWTCEvent), which would otherwise confuse pydantic's construction
            # path. The instance is retagged to `cls` once fully built.
            sev = GWEvent.model_validate(rep.json())
            if classification:
                _fetch_classification(sev, client)
            pe_fields = [
                "gw_id",
                "catalog",
                "gps",
                "far",
                "p_astro",
                "network_snr",
                "mass_1",
                "mass_2",
                "chirp_mass",
                "total_mass",
                "final_mass",
                "luminosity_distance",
                "redshift",
                "chi_eff",
                "preferred_waveform",
                "posterior_url",
                "strain_files",
            ]
            for field in pe_fields:
                val = getattr(gwosc, field)
                if val is not None:
                    setattr(sev, field, val)
            sev.__class__ = cls
        sev.save()
        return sev
    except OSError as e:
        cached = _load_cached_event(gw_name)
        if cached is None:
            raise
        print(f"Network unavailable ({e}); falling back to cached event for {gw_name}")
        cached.__class__ = cls
        return cached


def to_dataframe(events: list["GWEvent"]) -> pd.DataFrame:
    """Convert a list of GWEvents to a DataFrame, one row per event."""
    return pd.DataFrame([ev.to_dict() for ev in events])
