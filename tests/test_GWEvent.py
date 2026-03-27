from ligo.gracedb.rest import GraceDb
from GWUtils.models_gw import GWEvent, Detector
import datetime

client = GraceDb()

def test_validate_gw_event():
    sid_exs = ['S250328ae']

    for sid in sid_exs:
        (gwevent_data,) = client.superevents(sid) # To get rid of iterator, only one event expectd from query

        gwevent = GWEvent.model_validate(gwevent_data)
        assert gwevent.superevent_id == sid
        assert gwevent.created.tzinfo is not None
        assert isinstance(gwevent.t_start, datetime.datetime)
        assert isinstance(gwevent.t_0, datetime.datetime)
        assert isinstance(gwevent.t_end, datetime.datetime)
        assert gwevent.far > 0
        assert gwevent.detectors is not None
        assert all(isinstance(d, Detector) for d in gwevent.detectors)
        assert gwevent.skymap_ready is True
