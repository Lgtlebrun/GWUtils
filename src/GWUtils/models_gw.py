from pydantic import BaseModel, Field, field_validator, ConfigDict, PrivateAttr
from typing import List, Optional, ClassVar, Any
from pathlib import Path
import datetime
from enum import Enum
from astropy.time import Time
from urllib.request import urlretrieve
from .define import SKYMAP_FITS_DIRECTORY, EVENTS_DIRECTORY
from ligo.gracedb.rest import GraceDb
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

    # ── Identity ──────────────────────────────────────────────────────────────
    superevent_id: str
    gw_id: Optional[str] = None
    catalog: Optional[str] = None

    # ── Timing ────────────────────────────────────────────────────────────────
    created: Optional[datetime.datetime] = None
    t_start: Optional[datetime.datetime] = None
    t_end: Optional[datetime.datetime] = None
    t_0: Optional[datetime.datetime] = None
    gps: Optional[float] = None

    # ── Detectors & pipeline ──────────────────────────────────────────────────
    detectors: List[Detector] = []
    group: Optional[str] = None
    preferred_event: Optional[str] = None
    network_snr: Optional[float] = None

    # ── Rates ─────────────────────────────────────────────────────────────────
    far: Optional[float] = None
    p_astro: Optional[float] = None

    # ── Status flags ──────────────────────────────────────────────────────────
    skymap_ready: bool = False
    pastro_ready: bool = False

    # ── Classification ────────────────────────────────────────────────────────
    classification: Optional[CBCClassification] = None

    # ── Masses [M_sun] ────────────────────────────────────────────────────────
    mass_1: Optional[UncertainQuantity] = None
    mass_2: Optional[UncertainQuantity] = None
    chirp_mass: Optional[UncertainQuantity] = None
    total_mass: Optional[UncertainQuantity] = None
    final_mass: Optional[UncertainQuantity] = None

    # ── Distance & cosmology ──────────────────────────────────────────────────
    luminosity_distance: Optional[UncertainQuantity] = None  # [Mpc]
    redshift: Optional[UncertainQuantity] = None

    # ── Spins ─────────────────────────────────────────────────────────────────
    chi_eff: Optional[UncertainQuantity] = None

    # ── metadata ───────────────────────────────────────────────────────────
    preferred_waveform: Optional[str] = None
    posterior_url: Optional[str] = None
    strain_files: Optional[list[dict]] = None

    # ── Skymap ────────────────────────────────────────────────────────────────
    skymap_path: Optional[str | Path] = None
    _skymap: Optional[Any] = PrivateAttr(default=None)
    _meta: Optional[Any] = PrivateAttr(default=None)

    model_config: ClassVar = {"extra": "ignore"}

    skymap_priorities_map: ClassVar = {
        "CBC": ["gw", "Bilby", "bayestar"],
        "Burst": ["cwb", "cwb.LHV"],
    }

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

    def enrich_from_gwosc(self) -> "GWEvent":
        """
        Overlay catalog-quality GWOSC parameters onto a GraceDB-sourced event.
        Preserves GraceDB identity, timing, skymap status, and classification.
        """
        name = self.gw_id or self.superevent_id
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
        matching = list(Path(SKYMAP_FITS_DIRECTORY).glob(f"{self.superevent_id}*.fits"))
        if matching:
            return matching[0]
        return False

    def download_skymap(self):
        if self.skymap_ready:
            if self.group is not None:
                pipelines = GWEvent.skymap_priorities_map.get(self.group)
                if pipelines is not None:
                    dl_success = False
                    for pipeline in pipelines:
                        if pipeline == "gw":
                            pipeline = f"gw{self.superevent_id[1:]}_skymap"  # e.g. gw190425z_skymap
                        skymap_url = f"https://gracedb.ligo.org/api/superevents/{self.superevent_id}/files/{pipeline}.multiorder.fits"
                        filename = (
                            Path(SKYMAP_FITS_DIRECTORY)
                            / f"{self.superevent_id}_{pipeline}.fits"
                        )
                        try:
                            urlretrieve(skymap_url, filename)
                            print(f"Skymap downloaded successfully from: {skymap_url}")
                            dl_success = True
                            self.skymap_path = filename
                            return filename
                        except Exception as e:
                            print(
                                f"Failed to download skymap from: {skymap_url}. Error: {e}"
                            )
                        print(f"Attempting to download skymap from: {skymap_url}")
                    if not dl_success:
                        print(
                            f"Failed to download skymap from all pipelines for group {self.group}."
                        )
                else:
                    print(
                        f"No known skymap pipelines for group {self.group}. Cannot determine skymap URL."
                    )
            else:
                print("Group information is missing. Cannot determine skymap URL.")
        else:
            print("Skymap is not ready. Cannot download skymap.")

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

    def get_type_CBC(self):
        """Returns most probable type of CBC event, relying on classification"""
        if (not self.group == "CBC") | (self.classification is None):
            return "Unknown"

        elif self.classification is not None:
            label, prob = self.classification.most_probable()
            return label, prob

    def save(self):
        with open(EVENTS_DIRECTORY / f"{self.superevent_id}.json", "w") as file:
            file.write(self.model_dump_json())

    def plot_event(
        self,
        figPath: Path | str | None = None,
        circle_roi: bool = False,
        rect_roi: bool = False,
    ):
        skymap, meta = self.load_skymap()
        fig = plt.figure(figsize=(9, 4), dpi=100)
        ax = plt.axes(projection="astro hours mollweide")
        ax.grid()
        ax.imshow_hpx(skymap, cmap="cylon")

        if circle_roi:
            roi = self.get_90_roi_circle()
            circle = SphericalCircle(
                center=SkyCoord(roi["ra"] * u.deg, roi["dec"] * u.deg),
                radius=roi["radius_deg"] * u.deg,
                transform=ax.get_transform("icrs"),
                edgecolor="white",
                facecolor="none",
                linewidth=1.5,
                linestyle="--",
                label="90% CI",
            )
            ax.add_patch(circle)
            ax.legend(loc="lower right")
        if rect_roi:
            roi = self.get_90_roi_rect()
            ra_min, ra_max = roi["ra_min"], roi["ra_max"]
            dec_min, dec_max = roi["dec_min"], roi["dec_max"]

            # Draw the 4 edges as lines in icrs coordinates to handle projection correctly
            ra_top = np.linspace(ra_min, ra_max, 100)
            ra_bottom = np.linspace(ra_min, ra_max, 100)
            dec_left = np.linspace(dec_min, dec_max, 100)
            dec_right = np.linspace(dec_min, dec_max, 100)

            transform = ax.get_transform("icrs")
            kwargs = dict(
                transform=transform, color="cyan", linewidth=1.5, linestyle="--"
            )

            ax.plot(ra_top, np.full(100, dec_max), **kwargs)  # top
            ax.plot(ra_bottom, np.full(100, dec_min), **kwargs)  # bottom
            ax.plot(np.full(100, ra_min), dec_left, **kwargs)  # left
            ax.plot(
                np.full(100, ra_max), dec_right, **kwargs, label="90% bbox"
            )  # right

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
        cls, gw_name: str | list[str], client=None, classification: bool = True
    ):
        if isinstance(gw_name, list):
            return [
                _build_gwevent_from_gw_name(
                    name, client=client, classification=classification
                )
                for name in gw_name
            ]
        return _build_gwevent_from_gw_name(
            gw_name, client=client, classification=classification
        )


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
        files_dict = client.files(sev.preferred_event).json()
        for filename in files_dict:
            if filename.endswith(".json") and "p_astro" in filename:
                data = client.files(sev.preferred_event, filename).json()
                if is_classification_json(data):
                    sev.classification = CBCClassification.model_validate(data)
                    break
    except Exception as e:
        print(f"Could not fetch classification for {sev.superevent_id}: {e}")


def _build_gwevent(
    superevent_id: str, client=None, classification: bool = True
) -> GWEvent:
    if client is None:
        client = GraceDb()
    rep = client.superevent(superevent_id)
    sev = GWEvent.model_validate(rep.json())
    if classification:
        _fetch_classification(sev, client)
    return sev


def _build_gwevent_from_gw_name(
    gw_name: str, client=None, classification: bool = True
) -> GWEvent:
    if client is None:
        client = GraceDb()
    gwosc = GWEvent.from_gwosc(gw_name)
    rep = client.superevent(gwosc.superevent_id)
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
    return sev


def to_dataframe(events: list["GWEvent"]) -> pd.DataFrame:
    """Convert a list of GWEvents to a DataFrame, one row per event."""
    return pd.DataFrame([ev.to_dict() for ev in events])
