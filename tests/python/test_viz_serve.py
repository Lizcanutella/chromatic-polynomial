"""Serve-mode endpoint tests: real HTTP against a live server on an ephemeral port."""
import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest


@pytest.fixture(scope="module")
def server_url():
    from chromatic_rl.viz.serve import make_server
    srv = make_server(port=0, dataset="datasets/maps",
                      budget=5000, timeout=30.0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _server_has_maps() -> bool:
    from chromatic_rl.mapgraph import HAVE_MAPS
    return HAVE_MAPS


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return r.status, r.read()


def test_index_serves_viewer(server_url):
    with urllib.request.urlopen(server_url + "/") as r:
        html = r.read().decode()
    assert "VIZ_DOC" in html and '"mode": "serve"' in html.replace("'", '"')


def test_generate(server_url):
    with urllib.request.urlopen(
            server_url + "/generate?family=cycle&n=6") as r:
        d = json.loads(r.read())
    assert d["n"] == 6 and len(d["edges"]) == 6


def test_instances_and_instance(server_url):
    with urllib.request.urlopen(server_url + "/instances") as r:
        ids = json.loads(r.read())["ids"]
    assert ids
    with urllib.request.urlopen(server_url + f"/instance?id={ids[0]}") as r:
        d = json.loads(r.read())
    assert d["n"] >= 1 and isinstance(d["edges"], list)


def test_compute_roundtrip(server_url):
    import chromatic_rl as crl
    payload = {"n": 5, "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 2]],
               "selector": "min_degree_sum", "use_cache": False, "scores": False,
               "positions": [[0.1 * i, 0.5] for i in range(5)]}
    status, body = _post(server_url + "/compute", payload)
    d = json.loads(body)
    assert status == 200 and d["ok"], d
    doc = d["doc"]
    g = crl.Graph(5, [tuple(e) for e in payload["edges"]])
    ref = crl.compute(g, use_cache=False)
    assert doc["meta"]["poly"]["coeffs"] == [str(c) for c in ref.poly.coeffs()]
    assert doc["meta"]["mode"] == "serve"
    assert doc["layout"] == payload["positions"]
    assert doc["nodes"]


def test_compute_budget_exceeded(server_url):
    # dense-ish n=12 cache-OFF with a tiny budget must fail politely
    import itertools, random
    rng = random.Random(0)
    edges = [list(e) for e in itertools.combinations(range(12), 2) if rng.random() < 0.6]
    payload = {"n": 12, "edges": edges, "selector": "min_degree_sum",
               "use_cache": False, "scores": False, "budget": 3}
    status, body = _post(server_url + "/compute", payload)
    d = json.loads(body)
    assert status == 200 and not d["ok"]
    assert "budget" in d["error"].lower()


def test_compute_rejects_bad_selector(server_url):
    payload = {"n": 3, "edges": [[0, 1], [1, 2]], "selector": "nope",
               "use_cache": False, "scores": False}
    status, body = _post(server_url + "/compute", payload)
    d = json.loads(body)
    assert not d["ok"] and "selector" in d["error"].lower()


def test_freeze_returns_html(server_url):
    payload = {"n": 3, "edges": [[0, 1], [1, 2]], "selector": "none",
               "use_cache": False, "scores": False}
    _, body = _post(server_url + "/compute", payload)
    doc = json.loads(body)["doc"]
    status, html = _post(server_url + "/freeze", doc)
    assert status == 200
    assert b"VIZ_DOC" in html and b"/*__VIZ_DOC__*/" not in html


def test_index_contains_editor(server_url):
    with urllib.request.urlopen(server_url + "/") as r:
        html = r.read().decode()
    for marker in ("editor-canvas", "btn-compute", "mode-addv", "btn-freeze",
                   "up-file", "up-segment", "gal-select", "gal-load"):
        assert marker in html, marker


def test_index_contains_original_toggle_and_labels(server_url):
    with urllib.request.urlopen(server_url + "/") as r:
        html = r.read().decode()
    # backdrop toggles on both canvases
    assert 'id="up-original"' in html
    assert 'id="build-original"' in html
    # backdrop drawn as an <image> stretched to the position rect (exact overlay)
    assert "preserveAspectRatio" in html and "setBackdrop" in html
    # region-name tooltip on editor vertices
    assert 'append("title")' in html


# /segment runs in the SERVER, so the predicate is the server-side extra, not
# Pillow: Pillow alone is enough to build the fixture but not to segment it,
# which is exactly how these two tests failed rather than skipped.
requires_maps = pytest.mark.skipif(
    not _server_has_maps(),
    reason="needs the optional [maps] extra (pillow + scikit-image + scipy)")


def _png_bytes(arr):
    pytest.importorskip("PIL")
    import io
    from PIL import Image
    import numpy as np
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), "RGB").save(buf, format="PNG")
    return buf.getvalue()


@requires_maps
def test_segment_2x2_block(server_url):
    import base64
    import numpy as np
    a = np.zeros((60, 60, 3), np.uint8)
    a[:30, :30] = (220, 40, 40); a[:30, 30:] = (40, 160, 40)
    a[30:, :30] = (40, 40, 220); a[30:, 30:] = (200, 200, 40)
    payload = {"image": base64.b64encode(_png_bytes(a)).decode(),
               "style": "filled", "min_area_frac": 0.005, "sensitivity": 0.0}
    status, body = _post(server_url + "/segment", payload)
    d = json.loads(body)
    assert status == 200 and d["ok"], d
    mg = d["mapgraph"]
    assert mg["n"] == 4 and len(mg["edges"]) == 6      # K4
    assert len(mg["positions"]) == 4
    assert mg["detected_style"] == "filled"


@requires_maps
def test_segment_blank_no_regions(server_url):
    import base64
    import numpy as np
    blank = np.full((40, 40, 3), 255, np.uint8)        # all white → <2 regions
    payload = {"image": base64.b64encode(_png_bytes(blank)).decode(),
               "style": "filled"}
    status, body = _post(server_url + "/segment", payload)
    d = json.loads(body)
    assert status == 200 and not d["ok"]
    assert "no regions" in d["error"].lower()


def test_compute_large_doc_not_false_timeout(server_url):
    # Petersen cache-OFF yields a multi-hundred-node doc (>128KB JSON): the
    # old join-before-drain guard deadlocked on the queue pipe and falsely
    # reported a timeout. Must return ok fast.
    import networkx as nx
    import time
    G = nx.petersen_graph()
    edges = [list(e) for e in G.edges()]
    payload = {"n": 10, "edges": edges, "selector": "min_degree_sum",
               "use_cache": False, "scores": False}
    t0 = time.monotonic()
    status, body = _post(server_url + "/compute", payload)
    elapsed = time.monotonic() - t0
    d = json.loads(body)
    assert status == 200 and d["ok"], d
    assert len(d["doc"]["nodes"]) > 200
    assert elapsed < 20     # server fixture timeout is 30s; old bug took the full timeout


@pytest.fixture(scope="module")
def maps_server_url():
    # Serves the real, committed 216-instance maps dataset (datasets/maps/) — no
    # build pipeline needed, so this only needs the shipped data + [geo] extra absent.
    from chromatic_rl.viz.serve import make_server
    srv = make_server(port=0, maps_dataset="datasets/maps", budget=5000, timeout=30.0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_maps_list(maps_server_url):
    with urllib.request.urlopen(maps_server_url + "/maps") as r:
        d = json.loads(r.read())
    fams = {m["family"] for m in d["maps"]}
    assert fams == {"map_world", "map_country"}


def test_maps_list_carries_granularity_and_tier(maps_server_url):
    # Unit 7 + 8: the Gallery card needs the administrative granularity (to disclose
    # a states-vs-municipalities mismatch) and the tier (to warn on uncertified maps).
    with urllib.request.urlopen(maps_server_url + "/maps") as r:
        maps = json.loads(r.read())["maps"]
    countries = [m for m in maps if m["family"] == "map_country"]
    assert any(m["granularity"] for m in countries)   # at least one carries an admin level
    assert any(m["tier"] == "map" for m in countries)  # at least one is certified
    world = next(m for m in maps if m["family"] == "map_world")
    assert world["granularity"] is None          # worlds have no admin level


def test_map_by_id(maps_server_url):
    with urllib.request.urlopen(maps_server_url + "/maps") as r:
        first = json.loads(r.read())["maps"][0]
    with urllib.request.urlopen(
            maps_server_url + "/map?id=" + urllib.parse.quote(first["id"])) as r:
        m = json.loads(r.read())
    assert m["n"] == first["n"] and isinstance(m["edges"], list)
    assert len(m["positions"]) == m["n"] and isinstance(m["labels"], list)
    assert len(m["labels"]) == m["n"]                    # real region names, not a stub
    assert all(isinstance(s, str) for s in m["labels"])


def test_map_unknown_id_404(maps_server_url):
    try:
        urllib.request.urlopen(maps_server_url + "/map?id=nope")
        assert False, "expected 404"
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_map_has_original_false_without_geodata(maps_server_url):
    # No --geodata configured on this fixture -> /map advertises no original,
    # and /mapgeo declines rather than pretending.
    with urllib.request.urlopen(maps_server_url + "/maps") as r:
        first = json.loads(r.read())["maps"][0]
    with urllib.request.urlopen(
            maps_server_url + "/map?id=" + urllib.parse.quote(first["id"])) as r:
        assert json.loads(r.read())["has_original"] is False
    try:
        urllib.request.urlopen(
            maps_server_url + "/mapgeo?id=" + urllib.parse.quote(first["id"]))
        assert False, "expected 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        assert json.loads(e.read())["ok"] is False


def test_mapgeo_unknown_id_404(maps_server_url):
    try:
        urllib.request.urlopen(maps_server_url + "/mapgeo?id=nope")
        assert False, "expected 404"
    except urllib.error.HTTPError as e:
        assert e.code == 404


@pytest.fixture(scope="module")
def geo_maps_server_url():
    # End-to-end /mapgeo against the committed maps dataset + real Natural Earth
    # shapefiles. Skips where either the [geo] extra or the data files are absent.
    import os
    pytest.importorskip("geopandas")
    if not (os.path.isdir("datasets/maps") and os.path.isdir("datasets/naturalearth")):
        pytest.skip("committed maps dataset / Natural Earth data not present")
    # The committed artifact records which pipeline generation built it. An older
    # one cannot align with today's reconstruction (different key space, different
    # NE scale), and serve.py's guard correctly refuses. Skip rather than fail —
    # Phase B's rebuild stamps the current version and this un-skips itself.
    import json as _json
    from chromatic_rl.maps_geo.build import PIPELINE_VERSION
    with open("datasets/maps/manifest.json") as fh:
        _extra = _json.load(fh).get("extra", {})
    if _extra.get("pipeline") != PIPELINE_VERSION:
        pytest.skip(
            f"committed maps built by pipeline {_extra.get('pipeline')!r} "
            f"(data_source: {_extra.get('data_source')!r}); current is "
            f"{PIPELINE_VERSION!r} - rebuild (Phase B) required before outlines align")
    from chromatic_rl.viz.serve import make_server
    srv = make_server(port=0, maps_dataset="datasets/maps",
                      geodata="datasets/naturalearth", budget=5000, timeout=60.0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_mapgeo_roundtrip_aligns(geo_maps_server_url):
    from shapely.geometry import Point, Polygon
    with urllib.request.urlopen(geo_maps_server_url + "/maps") as r:
        maps = json.loads(r.read())["maps"]
    # Smallest country with substantial geometry (n>=10): avoids tiny island
    # nations where 0.02deg simplification erases sub-km atolls (a cosmetic
    # artifact; the keys==labels guard already guarantees correct mapping).
    cands = sorted((m for m in maps if m["family"] == "map_country" and m["n"] >= 10),
                   key=lambda m: m["n"])
    if not cands:
        pytest.skip("no mid-size country in maps dataset")
    tgt = cands[0]
    with urllib.request.urlopen(
            geo_maps_server_url + "/map?id=" + urllib.parse.quote(tgt["id"])) as r:
        mp = json.loads(r.read())
    assert mp["has_original"] is True
    with urllib.request.urlopen(
            geo_maps_server_url + "/mapgeo?id=" + urllib.parse.quote(tgt["id"])) as r:
        geo = json.loads(r.read())
    assert geo["ok"] is True
    assert len(geo["polygons"]) == mp["n"]
    # served centroids fall inside their own region outline (alignment proof);
    # allow a small slack for simplification nicking a coastal centroid.
    inside = sum(any(Polygon(r).buffer(1e-6).contains(Point(pos[0], pos[1]))
                     for r in rings if len(r) >= 4)
                 for pos, rings in zip(mp["positions"], geo["polygons"]))
    assert inside >= 0.9 * mp["n"]


# --- Unit 8: user timeout knob, budget default, timeout message ----------------

def test_effective_timeout_clamps_and_defaults():
    from chromatic_rl.viz.serve import _effective_timeout
    assert _effective_timeout(None, 60.0) == 60.0        # unset -> server default
    assert _effective_timeout(30, 60.0) == 30.0          # honoured
    assert _effective_timeout(99999, 60.0) == 600.0      # hard-capped (DoS guard)
    assert _effective_timeout(0, 60.0) == 60.0           # non-positive -> default
    assert _effective_timeout(-5, 60.0) == 60.0
    assert _effective_timeout("abc", 60.0) == 60.0       # unparseable -> default


def test_timeout_message_names_the_knob():
    from chromatic_rl.viz.serve import _timeout_message
    msg = _timeout_message(60.0)
    assert "60" in msg
    assert "timeout" in msg.lower()          # names the adjustable knob
    assert "600" in msg                      # states the hard maximum


def test_make_server_budget_default_matches_maps_cap():
    # A map certified at up to branch_cap=50000 must not fail in the viewer under a
    # smaller default; align the serve default with the maps dataset's cap.
    import inspect
    from chromatic_rl.viz.serve import make_server
    assert inspect.signature(make_server).parameters["budget"].default == 50000


def test_compute_honours_and_clamps_payload_timeout(server_url, monkeypatch):
    import chromatic_rl.viz.serve as serve
    seen = []
    def fake(payload, budget, timeout):
        seen.append(timeout)
        return {"ok": True, "doc": {}}
    monkeypatch.setattr(serve, "compute_with_guard", fake)
    base = {"n": 3, "edges": [[0, 1], [1, 2]]}
    _post(server_url + "/compute", {**base, "timeout": 45})
    _post(server_url + "/compute", {**base, "timeout": 99999})
    _post(server_url + "/compute", base)
    assert seen[0] == 45.0                    # honoured
    assert seen[1] == 600.0                   # clamped to the hard max
    assert seen[2] == 30.0                    # falls back to the server default
