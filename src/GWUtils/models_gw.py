from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, ClassVar
from pathlib import Path
import datetime
from enum import Enum
from astropy.time import Time
from urllib.request import urlretrieve
from .define import SKYMAP_FITS_DIRECTORY, EVENTS_DIRECTORY

class Detector(Enum):
    LIGO_Hanford = "H1"
    LIGO_Livingston = "L1"
    Virgo = "V1"
    KAGRA = "K1"

    def __str__(self):
        return self.name

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

    superevent_id: str
    gw_id: Optional[str]
    created: datetime.datetime # Needs timestamp to datetime
    t_start: datetime.datetime # Needs GPS to datetime
    t_end: datetime.datetime # Needs GPS to datetime
    t_0: Optional[datetime.datetime] # Needs GPS to datetime
    skymap_ready: bool  # Needs to process the labels entry to check SKYMAP_READY
    far : float
    detectors: List[Detector]
    group : Optional[str]
    preferred_event : str
    skymap : Optional[str]

    model_config : ClassVar = {
        "extra": "ignore",  # Ignore extra fields from JSON
    } 

    skymap_priorities_map : ClassVar = {
        'CBC' : ['Bilby', 'bayestar'],
        'Burst' : ['cwb', 'cwb.LHV']
    }

    @field_validator('detectors', mode='before')
    def validate_detectors(cls, v):
        if isinstance(v, list):
            return [Detector(det) for det in v]
        elif isinstance(v, str):
            return [Detector(det) for det in v.split(',')]
        raise ValueError("Detectors must be a list of strings or a string of comma-separated values")
    
    @field_validator('created', mode='before')
    @classmethod
    def parse_created(cls, v):
        """Convert '2025-11-17 21:38:45 UTC' to datetime."""
        if isinstance(v, str):
            # ISO format from model dump: '2025-03-28T05:40:42Z'
            if 'T' in v:
                return datetime.datetime.fromisoformat(v.replace('Z', '+00:00'))
            dt_str = v.replace(' UTC', '').strip()
            dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
            return dt.replace(tzinfo=datetime.timezone.utc)
        return v
    
    @field_validator('t_start', 't_end', 't_0', mode='before')
    @classmethod
    def parse_gps_time(cls, v):
        """Convert GPS time to datetime."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            t = Time(v, format='gps', scale='utc')
            return t.to_datetime(timezone=datetime.timezone.utc)
        return v
    
    
    @classmethod
    def model_validate(cls, obj, **kwargs):
        """Extract nested fields and process labels."""
        obj = dict(obj)  # Make a copy to modify

        # If already a flat model dump, skip GraceDB-specific extraction
        if 'preferred_event_data' not in obj:
            return super().model_validate(obj, **kwargs)
    
        # Extract labels
        labels = obj.get('labels', [])
        obj['labels'] = labels
        
        # Extract instruments from preferred_event_data
        preferred = obj.get('preferred_event_data', {})
        if preferred:
            obj['detectors'] = preferred.get('instruments')
            obj['group'] = preferred.get('group')
            obj["preferred_event"] = preferred.get('graceid')
        
        obj['skymap_ready'] = 'SKYMAP_READY' in labels or 'SKYMAP_READY' in preferred.get('labels', [])

        # If skymap downloaded, fetch the filename
        skymap_path = Path(SKYMAP_FITS_DIRECTORY) / f"{obj['superevent_id']}*.fits"
        matching_files = list(Path(SKYMAP_FITS_DIRECTORY).glob(f"{obj['superevent_id']}_*.fits"))
        obj['skymap'] = str(matching_files[0]) if matching_files else None

        return super().model_validate(obj, **kwargs)

    def download_skymap(self):
        if self.skymap_ready:
            if self.group is not None :
                pipelines = GWEvent.skymap_priorities_map.get(self.group)
                if pipelines is not None:
                    dl_success = False
                    for pipeline in pipelines:
                        skymap_url = f"https://gracedb.ligo.org/api/superevents/{self.superevent_id}/files/{pipeline}.multiorder.fits"
                        filename = Path(SKYMAP_FITS_DIRECTORY) / f"{self.superevent_id}_{pipeline}.fits"
                        try:
                            urlretrieve(skymap_url, filename)
                            print(f"Skymap downloaded successfully from: {skymap_url}")
                            dl_success = True
                            return filename
                        except Exception as e:
                            print(f"Failed to download skymap from: {skymap_url}. Error: {e}")
                        print(f"Attempting to download skymap from: {skymap_url}")
                    if not dl_success:
                        print(f"Failed to download skymap from all pipelines for group {self.group}.")
                else:
                    print(f"No known skymap pipelines for group {self.group}. Cannot determine skymap URL.")
            else:
                print("Group information is missing. Cannot determine skymap URL.")
        else:
            print("Skymap is not ready. Cannot download skymap.")
            
    def save(self):
        with open(EVENTS_DIRECTORY / f"{self.superevent_id}.json", 'w') as file :
            file.write(self.model_dump_json())

    def get_classification(self):

if __name__ == "__main__":

    example = {
        'superevent_id': 'S251117dq',
        'gw_id': None,
        'category': 'Production',
        'created': '2025-11-17 21:38:45 UTC',
        't_start': 1447450730.097656,
        't_0': 1447450731.119385,
        't_end': 1447450732.121005,
        'far': 5.867636084251777e-15,
        'labels': ['EM_READY', 'PE_READY', 'SKYMAP_READY'],
        'preferred_event_data': {
            'instruments': 'H1,L1',
        }
    }
        
    # Parse it
    event : GWEvent = GWEvent.model_validate(example)

    # Access the fields
    print(f"Event ID: {event.superevent_id}")
    print(f"Created: {event.created}")
    print(f"Coalescence: {event.t_0}")
    print(f"FAR: {event.far:.2e} Hz")
    print(f"Skymap ready: {event.skymap_ready}")
    print(f"Detectors: {event.detectors}")
    print(f"Duration: {(event.t_end - event.t_start).total_seconds():.3f} s")
