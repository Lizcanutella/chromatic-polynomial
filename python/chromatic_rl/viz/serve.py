"""Localhost serve mode: graph editor + live compute.

Compute runs in a FRESH subprocess per request (multiprocessing spawn of a
module-level worker): (a) a wall-clock timeout can kill a stuck run, the C++
engine is not interruptible in-process; (b) cache-ON runs get private nauty
globals, so a stray concurrent request cannot corrupt the canonicalizer.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import queue as _queue
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _compute_worker(payload, budget, q):
    """Runs in a child process; puts {'ok': bool, ...} on q."""
    import chromatic_rl as crl
    from chromatic_rl.dataset.score import resolve_selector
    from chromatic_rl.viz.doc import build_doc

    try:
        n = int(payload["n"])
        edges = [tuple(e) for e in payload["edges"]]
        g = crl.Graph(n, edges)
        sel_name = payload.get("selector", "none")
        sel = resolve_selector(sel_name)
        use_cache = bool(payload.get("use_cache", True))
        res = crl.compute(g, selector=sel, use_cache=use_cache, record="full",
                          branch_budget=int(payload.get("budget") or budget))
        doc = build_doc(
            res, n, edges, selector_name=sel_name, use_cache=use_cache,
            mode="serve", positions=payload.get("positions"),
            want_scores=bool(payload.get("scores", False)),
            input_desc=payload.get("desc", f"drawn graph n={n} m={len(edges)}"))
        q.put({"ok": True, "doc": doc})
    except crl.BudgetExceeded:
        q.put({"ok": False,
               "error": "branch budget exceeded. Try a smaller/sparser graph, "
                        "or enable the cache"})
    except Exception as e:  # noqa: BLE001 - surfaced to the UI
        q.put({"ok": False, "error": f"{type(e).__name__}: {e}"})


# A user-settable /compute timeout is a DoS lever on a publicly-hosted server
# (each call spawns a subprocess), so it is hard-capped regardless of request.
MAX_TIMEOUT_S = 600.0


def _effective_timeout(requested, default):
    """Resolve the per-request timeout: honour a positive numeric request up to
    MAX_TIMEOUT_S; fall back to the server default for anything unset, non-positive
    or unparseable."""
    if requested is None:
        return float(default)
    try:
        t = float(requested)
    except (TypeError, ValueError):
        return float(default)
    if t <= 0:
        return float(default)
    return min(t, MAX_TIMEOUT_S)


def _timeout_message(timeout):
    return (f"timed out after {timeout:.0f}s. The graph is too large to finish "
            f"in the time limit; raise the timeout (up to {MAX_TIMEOUT_S:.0f} s) "
            f"in Build, or use a smaller/sparser graph.")


def _guarded(target, args, timeout):
    # forkserver, not the platform default (fork): this handler runs on a
    # ThreadingHTTPServer worker thread, and forking from a multi-threaded
    # process is deprecated (Python 3.12+) and a real deadlock hazard.
    ctx = mp.get_context("forkserver")
    q = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, q))
    proc.start()
    deadline = time.monotonic() + timeout
    out = None
    while time.monotonic() < deadline:
        try:
            out = q.get(timeout=0.1)     # draining the pipe unblocks the child
            break
        except _queue.Empty:
            if not proc.is_alive():      # died without reporting
                break
    if out is None:
        if proc.is_alive():
            out = {"ok": False, "error": _timeout_message(timeout)}
        else:
            out = {"ok": False, "error": "worker crashed"}
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
    proc.join()
    return out


def compute_with_guard(payload, budget, timeout):
    return _guarded(_compute_worker, (payload, budget), timeout)


def _segment_worker(payload, q):
    """Runs in a child process; puts {'ok': bool, ...} on q."""
    import base64
    try:
        from chromatic_rl.mapgraph import image_to_graph, MapGraphUnavailable
    except Exception as e:  # noqa: BLE001
        q.put({"ok": False, "error": f"map upload unavailable: {e}"})
        return
    try:
        img = base64.b64decode(payload["image"])
        mg = image_to_graph(
            img, style=payload.get("style", "auto"),
            min_area_frac=float(payload.get("min_area_frac", 0.005)),
            sensitivity=float(payload.get("sensitivity", 0.0)))
    except MapGraphUnavailable as e:
        q.put({"ok": False, "error": str(e)})
        return
    except Exception as e:  # noqa: BLE001 - surfaced to the UI
        q.put({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return
    if mg.n < 2:
        q.put({"ok": False,
               "error": "no regions detected - adjust min area / style"})
        return
    q.put({"ok": True, "mapgraph": {
        "n": mg.n, "edges": [list(e) for e in mg.edges],
        "positions": mg.positions, "regions": mg.regions,
        "detected_style": mg.detected_style}})


def segment_with_guard(payload, timeout):
    return _guarded(_segment_worker, (payload,), timeout)


def _outlines_worker(payload, q):
    """Runs in a child process; reconstructs a map instance's region borders."""
    try:
        from chromatic_rl.maps_geo import MapsGeoUnavailable
        from chromatic_rl.maps_geo.outlines import region_outlines
    except Exception as e:  # noqa: BLE001
        q.put({"ok": False, "error": f"map outlines unavailable: {e}"})
        return
    try:
        keys, polygons = region_outlines(
            payload["geodata"], family=payload["family"],
            iso=payload.get("iso"), policy=payload.get("policy"))
    except MapsGeoUnavailable as e:
        q.put({"ok": False, "error": str(e)})
        return
    except Exception as e:  # noqa: BLE001 - surfaced to the UI
        q.put({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return
    # Alignment guard: the reconstructed unit order must match the dataset's
    # labels, or the overlaid borders would silently sit on the wrong vertices.
    expected = payload.get("labels") or []
    if expected and keys != expected:
        q.put({"ok": False,
               "error": "geodata no longer matches the committed dataset "
                        "(region set changed); outlines would misalign"})
        return
    q.put({"ok": True, "polygons": polygons})


def outlines_with_guard(payload, timeout):
    return _guarded(_outlines_worker, (payload,), timeout)


def _make_handler(cfg):
    from chromatic_rl.viz.export import render

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, html, download=False):
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            if download:
                self.send_header("Content-Disposition",
                                 "attachment; filename=chromatic-run.html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(url.query)
            if url.path == "/":
                empty = {"meta": {"mode": "serve", "selector": "none",
                                  "use_cache": True, "scores": False,
                                  "input": {"desc": "", "n": 0, "edges": []},
                                  "stats": {}, "poly": None},
                         "layout": [], "nodes": []}
                return self._html(render(empty))
            if url.path == "/generate":
                return self._generate(qs)
            if url.path == "/instances":
                if not cfg["instances"]:
                    return self._json({"ids": []})
                return self._json({"ids": [i.id for i in cfg["instances"]]})
            if url.path == "/instance":
                iid = (qs.get("id") or [""])[0]
                inst = next((i for i in (cfg["instances"] or []) if i.id == iid), None)
                if inst is None:
                    return self._json({"error": f"unknown instance {iid!r}"}, 404)
                return self._json({"n": inst.n,
                                   "edges": [list(e) for e in inst.graph.edges()]})
            if url.path == "/maps":
                maps = cfg.get("maps") or []
                return self._json({"maps": [
                    {"id": i.id, "family": i.family, "n": i.n,
                     "label": i.params.get("iso", "world"),
                     "granularity": i.params.get("granularity"),  # Unit 7
                     "tier": i.tier,                               # Unit 8: warn if uncertified
                     "excluded_reason": i.excluded_reason}
                    for i in maps]})
            if url.path == "/map":
                iid = (qs.get("id") or [""])[0]
                inst = next((i for i in (cfg.get("maps") or []) if i.id == iid), None)
                if inst is None:
                    return self._json({"error": f"unknown map {iid!r}"}, 404)
                return self._json({
                    "n": inst.n,
                    "edges": [list(e) for e in inst.graph.edges()],
                    "positions": inst.positions,
                    "labels": inst.labels,
                    "has_original": bool(cfg.get("geodata"))})
            if url.path == "/mapgeo":
                iid = (qs.get("id") or [""])[0]
                inst = next((i for i in (cfg.get("maps") or []) if i.id == iid), None)
                if inst is None:
                    return self._json({"ok": False, "error": f"unknown map {iid!r}"}, 404)
                if not cfg.get("geodata"):
                    return self._json({"ok": False,
                                       "error": "no geodata source configured "
                                                "(pass --geodata)"}, 400)
                payload = {"geodata": cfg["geodata"], "family": inst.family,
                           "iso": inst.params.get("iso"),
                           "policy": inst.params.get("policy"),
                           "labels": inst.labels}
                out = outlines_with_guard(payload, cfg["timeout"])
                return self._json(out, 200 if out.get("ok") else 400)
            self.send_error(404)

        def _generate(self, qs):
            from chromatic_rl.bench import GENERATORS
            fam = (qs.get("family") or [""])[0]
            if fam not in GENERATORS:
                return self._json({"error": f"unknown family {fam!r}"}, 400)
            def gi(k, d):
                return int((qs.get(k) or [d])[0])
            try:
                if fam == "erdos_renyi":
                    g, _ = GENERATORS[fam](n=gi("n", 10),
                                           p=float((qs.get("p") or [0.3])[0]),
                                           seed=gi("seed", 1))
                elif fam == "random_regular":
                    g, _ = GENERATORS[fam](n=gi("n", 10), d=gi("d", 3),
                                           seed=gi("seed", 1))
                elif fam == "grid":
                    g, _ = GENERATORS[fam](rows=gi("rows", 3), cols=gi("cols", 4))
                elif fam == "random_tree":
                    g, _ = GENERATORS[fam](n=gi("n", 10), seed=gi("seed", 1))
                else:
                    g, _ = GENERATORS[fam](n=gi("n", 10))
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 400)
            return self._json({"n": g.num_vertices(),
                               "edges": [list(e) for e in g.edges()]})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return self._json({"ok": False, "error": "bad JSON"}, 400)
            if self.path == "/compute":
                # Unit 8: honour a user-supplied timeout, hard-capped at 600 s.
                timeout = _effective_timeout(payload.get("timeout"), cfg["timeout"])
                out = compute_with_guard(payload, cfg["budget"], timeout)
                return self._json(out)
            if self.path == "/segment":
                out = segment_with_guard(payload, cfg["timeout"])
                return self._json(out)
            if self.path == "/freeze":
                doc = dict(payload)
                doc.setdefault("meta", {})["mode"] = "export"
                return self._html(render(doc), download=True)
            self.send_error(404)

    return Handler


def make_server(port=8765, dataset=None, maps_dataset=None, geodata=None,
                budget=50000, timeout=60.0):
    instances = None
    if dataset:
        from chromatic_rl.dataset import load
        _manifest, instances = load(dataset)
    maps = None
    if maps_dataset:
        from chromatic_rl.dataset import load
        _mm, maps = load(maps_dataset)
    cfg = {"budget": budget, "timeout": timeout, "instances": instances,
           "maps": maps, "geodata": geodata}
    return ThreadingHTTPServer(("127.0.0.1", port), _make_handler(cfg))


def serve(port=8765, dataset=None, maps_dataset=None, geodata=None,
          budget=50000, timeout=60.0, open_browser=True):
    srv = make_server(port=port, dataset=dataset, maps_dataset=maps_dataset,
                      geodata=geodata, budget=budget, timeout=timeout)
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    print(f"viz server on {url}  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
