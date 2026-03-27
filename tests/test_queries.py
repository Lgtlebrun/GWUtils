from GWUtils.query_utils import query_cbc, query_superevent, validate_CBC

def test_query():
    sid = 'S250328ae'
    ss = query_superevent(sid)

    assert len(ss) == 1
    assert ss[0].superevent_id == sid # Simple check


def test_query_cbc():
    sid = 'S250328ae'
    ss = query_cbc(sid)

    print(f"DEBUG {ss[0].classification}")
    assert len(ss) == 1
    for ev in ss :
        assert ev.group == 'CBC'

# def test_query_many_cbc():
#     query = 'far < 1e-9 label: SKYMAP_READY'
#     ss = query_cbc(query)

#     assert len(ss) > 1
#     for ev in ss :
#         assert ev.group == 'CBC'