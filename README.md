# GWUtils
### A helper library for working with GW events

## Rationale 
The GW events data is scattered among human-vetted catalogs such as GWTC (mde avilable via the gwosc package) and wide-scoped databases such as GraceDB. This library incorporates several APIs into data fetching utilities, allowing ergonomic access to the data.

## Installation
From source :
- Clone this repository to a local directory
- Create environment (e.g., ```python3 -m venv .venv```)
- Run ```pip install .```, optionally ```pip install -e .``` for development

## Usage
The most important input of this library resides in the classes provided under ```GWutils.models_gw```.

### The GWEvent model
This model incorporates all the fetching logic from GraceDB. While it can be instantiated from a dictionary, the most convenient way to create such objects is to parse them via the qeury utils under ```query_utils.py```. For example, to query GraceDB for CBC events :

```python
from GWUtils.query_utils import query_cbc
events = query_cbc('far < 4')
```
will yield a list of corresponding GWEvent objects.

You can now load the latest Healpix skymap of a given event :
```python
event = events[0]
skymap, metadata = event.load_skymap()
```

Plot it :
```python
event.plot_event()
```

or convert the event to a one-row dataframe :

```python
df = event.to_dataframe()
```

If you are in possession of a list of events, i.e. : (see next section for description of the child GWTCEvent model)

```python
from GWUtils.models_gw import GWTCEvent
events = ["GW230627_015337", "GW230919_215712", "GW230922_020344", "GW231206_233901", "GW231226_101520"]
gwevents = [GWTCEvent(ev) for ev in events]
```
N.B. : GWUtils will download skymaps locally to your computer when not already available, and will preferentially fetch the skymap from there if existing.

you can also create a multi-rows dataframe directly via :
```python
from GWUtils.models_gw import to_dataframe
df = to_dataframe(gwevents)
```


### Events from GWTC
The ```gwosc``` package allows to fetch from the GWOSC database, especially from the GWTC catalog. 

For a given human-vetted GW event id, e.g. *GW231226_101520*, you can instantiate a ```GWTCEvent``` that inherits from ```GWEvent``` :
```python
from GWUtils.models_gw import GWTCEvent

event_name = "GW231226_101520"
event = GWTCEvent(event_name)
```

The GWOSC database is then fetched to obtain the latest CBC parameters posteriors, and the corresponding GraceDB superevent id. The latest skymap along with other metadata are then fetched from graceDB, and all thisinfo is stored in a single structure, the `event` object.
