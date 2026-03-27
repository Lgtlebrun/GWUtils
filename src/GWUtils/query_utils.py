from ligo.gracedb.rest import GraceDb
from .models_gw import GWEvent, CBCClassification, is_classification_json
from urllib import request
import re
from collections import defaultdict
from gwosc.api import fetch_event_json
from gwosc import datasets

def suffix_sort_key(suffix: str) -> tuple[int, str]:
    """
    LIGO suffixes go a, b, ..., z, aa, ab, ..., zz, aaa, ...
    Sort by length first (longer = later), then alphabetically.
    """
    return (len(suffix), suffix)

def filter_latest_events(events: list[GWEvent]) -> list[GWEvent]:
    """Keep only the latest event (highest suffix) for each date group."""
    grouped = defaultdict(list)
    for event in events:
        match = re.match(r'(S\d{6})([a-z]+)$', event.superevent_id)
        if match:
            prefix, suffix = match.group(1), match.group(2)
            grouped[prefix].append((suffix, event))
        else:
            # No suffix or unexpected format — keep as-is
            grouped[event.superevent_id].append(('', event))

    latest_events = []
    for prefix, suffix_event_list in grouped.items():
        _, best = max(suffix_event_list, key=lambda x: suffix_sort_key(x[0]))
        latest_events.append(best)

    return latest_events

def query_superevent(query, client = None) -> list[GWEvent]:
    if client is None:
        client = GraceDb()
    
    rep = client.superevents(query)
    superevents = [GWEvent.model_validate(ev) for ev in rep]

    return filter_latest_events(superevents)

def validate_CBC(sev : GWEvent):
    # Check there is a group
    if not sev.group :
        return False
    
    return sev.group == 'CBC'

def gw_name_to_superevent_id(gw_name: str, client=None) -> str:
    """Resolve a GW catalog name to a GraceDB superevent ID via GWOSC."""
    ev = fetch_event_json(gw_name)
    events = ev.get("events", {})
    if not events:
        raise ValueError(f"No GWOSC data found for {gw_name}")
    latest = max(events.values(), key=lambda e: e.get("version", 0))
    gracedb_id = latest.get("gracedb_id")
    if not gracedb_id:
        raise ValueError(f"No GraceDB ID found in GWOSC data for {gw_name}")
    return gracedb_id

def query_cbc(query: str, client=None, classification: bool = True, enrich: bool = True) -> list[GWEvent]:
    """
    Query GraceDB for CBC superevents.

    - Accepts any GraceDB query string (superevent ID, time range, label, etc.)
      or a GW catalog name (e.g. 'GW230627_015337').
    - Retains only CBC events.
    - Optionally fetches p_astro classification from GraceDB.
    - Optionally enriches all events with GWOSC catalog parameters (best-effort,
      silently skipped if the event is not in any catalog).
    """
    if client is None:
        client = GraceDb()

    # If a GW catalog name, resolve to superevent ID first
    if query.startswith("GW"):
        query = gw_name_to_superevent_id(query)

    superevents = query_superevent(query, client=client)
    superevents = [sev for sev in superevents if validate_CBC(sev)]

    if classification:
        for sev in superevents:
            if sev.pastro_ready:
                try:
                    files_dict = client.files(sev.preferred_event).json()
                    for filename in files_dict.keys():
                        if filename.endswith(".json") and 'p_astro' in filename:
                            data = client.files(sev.preferred_event, filename).json()
                            if is_classification_json(data):
                                sev.classification = CBCClassification.model_validate(data)
                                break
                except Exception as e:
                    print(f"Could not fetch classification for {sev.superevent_id}: {e}")

    if enrich:
        for sev in superevents:
            sev.enrich_from_gwosc()  # silent no-op if not in any catalog

    return superevents

def query_latest_gwtc(query: str = None) -> list[str]:
    all_datasets = datasets.find_datasets(type='event')
    gw_datasets = [gw for gw in all_datasets if gw.startswith('GW')]
    
    # If a query is provided, filter to datasets matching the event name
    if query is not None:
        gw_datasets = [gw for gw in gw_datasets if gw.startswith(query)]

    def _keep_latest_versions(items):
        latest = {}
        for item in items:
            parts = item.rsplit('-v', 1)
            if len(parts) == 2 and parts[1].isdigit():
                name, version = parts[0], int(parts[1])
                if name not in latest or version > latest[name]:
                    latest[name] = version
            else:
                latest[item] = None
        return [
            f"{name}-v{ver}" if ver is not None else name
            for name, ver in latest.items()
        ]

    return _keep_latest_versions(gw_datasets)