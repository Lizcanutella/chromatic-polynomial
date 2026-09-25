"""Image → graph pipeline (Unit B of the map-aware visualizer).

Segments a map image into regions and their adjacency graph, emitting the
project-standard (n, edges, positions) representation. Depends on the optional
``[maps]`` extra (Pillow + scikit-image + scipy); import-guarded so the core
package and /compute keep working without it.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

try:
    import numpy as np
    from PIL import Image
    import scipy.ndimage as ndi
    from skimage import measure
    HAVE_MAPS = True
except Exception:  # noqa: BLE001 - any import failure means the extra is absent
    HAVE_MAPS = False


class MapGraphUnavailable(RuntimeError):
    """Raised when the [maps] extra is not installed."""


@dataclass
class MapGraph:
    n: int
    edges: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    detected_style: str = "filled"


def _require_maps() -> None:
    if not HAVE_MAPS:
        raise MapGraphUnavailable(
            "the 'maps' extra is required for image upload - "
            "install it with:  uv pip install pillow scikit-image scipy")


def _load_rgb(image_bytes, max_side=1000):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                         Image.BILINEAR)
    return np.asarray(img)


def _detect_style(rgb):
    lum = rgb.mean(axis=2)
    dark = float((lum < 60).mean())
    white = float((lum > 200).mean())
    return "outline" if (dark < 0.25 and white > 0.5) else "filled"


def _segment_filled(rgb, n_colors=8, min_area_frac=0.005):
    q = Image.fromarray(rgb).convert("RGB").quantize(
        colors=n_colors, method=Image.Quantize.MEDIANCUT)
    idx = np.asarray(q)
    h, w = idx.shape
    total = h * w
    labels = np.full((h, w), -1, dtype=np.int64)
    next_id = 0
    struct = np.ones((3, 3), dtype=bool)          # 8-connectivity
    for pal in np.unique(idx):
        comp, ncomp = ndi.label(idx == pal, structure=struct)
        for c in range(1, ncomp + 1):
            if int((comp == c).sum()) >= min_area_frac * total:
                labels[comp == c] = next_id
                next_id += 1
    return labels


def _expand_labels(labels):
    bg = labels < 0
    if not bg.any():
        return labels
    _, (iy, ix) = ndi.distance_transform_edt(bg, return_indices=True)
    return labels[iy, ix]


def _adjacency(labels, n, sensitivity=0.0):
    if n < 2:
        return []
    h, w = labels.shape
    min_shared = max(1, round(sensitivity * min(h, w)))
    from collections import defaultdict
    contact = defaultdict(int)
    # 4 unique offsets cover all 8-connectivity neighbour pairs once each.
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        ay0, ay1 = max(0, -dy), h - max(0, dy)
        ax0, ax1 = max(0, -dx), w - max(0, dx)
        by0, by1 = max(0, dy), h - max(0, -dy)
        bx0, bx1 = max(0, dx), w - max(0, -dx)
        a = labels[ay0:ay1, ax0:ax1]
        b = labels[by0:by1, bx0:bx1]
        m = a != b
        u = np.minimum(a[m], b[m])
        v = np.maximum(a[m], b[m])
        valid = u >= 0
        if not valid.any():
            continue
        keys = u[valid].astype(np.int64) * n + v[valid].astype(np.int64)
        ids, cnts = np.unique(keys, return_counts=True)
        for k, c in zip(ids, cnts):
            contact[(int(k // n), int(k % n))] += int(c)
    return sorted(e for e, c in contact.items() if c >= min_shared)


def _region_props(labels, rgb, n):
    h, w = labels.shape
    out = []
    for i in range(n):
        mask = labels == i
        cy, cx = ndi.center_of_mass(mask)
        rgb_mean = rgb[mask].mean(axis=0)
        contours = measure.find_contours(mask.astype(float), 0.5)
        outline = []
        if contours:
            c = max(contours, key=len)
            step = max(1, len(c) // 24)
            outline = [[float(x) / w, float(y) / h] for y, x in c[::step]]
        out.append({"id": i,
                    "centroid": [float(cx) / w, float(cy) / h],
                    "rgb": [int(round(v)) for v in rgb_mean],
                    "area": int(mask.sum()),
                    "outline": outline})
    return out


def _segment_outline(rgb, min_area_frac=0.005):
    lum = rgb.mean(axis=2)
    regions_mask = lum >= 100                      # non-line (blank) areas
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)  # 4-connectivity
    comp, ncomp = ndi.label(regions_mask, structure=cross)
    h, w = lum.shape
    total = h * w
    labels = np.full((h, w), -1, dtype=np.int64)
    next_id = 0
    for c in range(1, ncomp + 1):
        if int((comp == c).sum()) >= min_area_frac * total:
            labels[comp == c] = next_id
            next_id += 1
    return labels


def image_to_graph(image_bytes, *, style="auto", min_area_frac=0.005,
                   sensitivity=0.0, n_colors=8, max_side=1000) -> MapGraph:
    _require_maps()
    rgb = _load_rgb(image_bytes, max_side)
    detected = _detect_style(rgb) if style == "auto" else style
    if detected == "outline":
        labels = _segment_outline(rgb, min_area_frac)     # defined in Task 4
    else:
        labels = _segment_filled(rgb, n_colors, min_area_frac)
    n = int(labels.max()) + 1 if (labels >= 0).any() else 0
    regions = _region_props(labels, rgb, n)
    edges = _adjacency(_expand_labels(labels), n, sensitivity)
    return MapGraph(n=n, edges=edges,
                    positions=[r["centroid"] for r in regions],
                    regions=regions, detected_style=detected)
