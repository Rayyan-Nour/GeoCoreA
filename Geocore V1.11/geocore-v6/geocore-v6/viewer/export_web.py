"""
Export pipeline outputs to a single self-contained interactive HTML map.

Replaces the VTK desktop viewer for day-to-day target review. The result is
one .html file that opens in any browser, can be e-mailed to a client, and
needs no installation:

* Leaflet pan/zoom map (CDN with offline-degradation notice)
* Hillshaded DEM base layer rendered from the run's own DEM
* Toggleable overlays: prospectivity probability, model uncertainty,
  target classes, spectral depth, Euler depth
* Per-layer opacity sliders and colorbar legends
* Click anywhere to read the underlying values (probability, uncertainty,
  depth) from an embedded downsampled value grid
* Deposit markers from the training CSV, if provided

Usage:
    from viewer.export_web import export_viewer
    export_viewer(run_dir, out_html)
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from pyproj import Transformer

try:
    import matplotlib
    _CMAPS = matplotlib.colormaps
except ImportError:  # pragma: no cover
    _CMAPS = None
from PIL import Image


# ----------------------------------------------------------------------
# Raster -> PNG helpers
# ----------------------------------------------------------------------

def _read(path: Path) -> Tuple[np.ndarray, "rasterio.Affine", str]:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        return data, src.transform, str(src.crs)


def _bounds_wgs84(shape: Tuple[int, int], transform, crs: str
                  ) -> List[List[float]]:
    """[[south, west], [north, east]] for Leaflet imageOverlay."""
    H, W = shape
    xs = [transform.c, transform.c + W * transform.a]
    ys = [transform.f, transform.f + H * transform.e]
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = [], []
    for x in xs:
        for y in ys:
            lon, lat = tr.transform(x, y)
            lons.append(lon)
            lats.append(lat)
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def _colormap_png(data: np.ndarray, cmap_name: str,
                  vmin: Optional[float] = None,
                  vmax: Optional[float] = None,
                  max_px: int = 1400) -> Tuple[str, float, float]:
    """Render array to a base64 PNG data URI. NaN -> transparent."""
    d = data.copy()
    finite = np.isfinite(d)
    if not finite.any():
        d = np.zeros_like(d)
        finite = np.ones_like(d, bool)
    lo = float(np.nanpercentile(d, 2)) if vmin is None else vmin
    hi = float(np.nanpercentile(d, 98)) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1.0

    # Downsample very large rasters for a reasonable file size
    H, W = d.shape
    step = max(1, int(np.ceil(max(H, W) / max_px)))
    d = d[::step, ::step]
    finite = finite[::step, ::step]

    norm = np.clip((d - lo) / (hi - lo), 0, 1)
    norm[~finite] = 0.0
    if _CMAPS is not None:
        rgba = (_CMAPS[cmap_name](norm) * 255).astype(np.uint8)
    else:  # grayscale fallback
        g = (norm * 255).astype(np.uint8)
        rgba = np.stack([g, g, g, np.full_like(g, 255)], axis=-1)
    rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return uri, lo, hi


def _hillshade(dem: np.ndarray, transform,
               azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    """Classic Horn hillshade for the base layer."""
    z = dem.copy()
    z[~np.isfinite(z)] = np.nanmedian(z[np.isfinite(z)]) if np.isfinite(z).any() else 0
    dx = abs(transform.a) or 1.0
    dy = abs(transform.e) or 1.0
    gy, gx = np.gradient(z, dy, dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    hs = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0, 1)


def _value_grid(data: np.ndarray, max_px: int = 220) -> Dict:
    """Downsampled grid embedded for click-query (kept small)."""
    H, W = data.shape
    step = max(1, int(np.ceil(max(H, W) / max_px)))
    sub = data[::step, ::step]
    sub = np.where(np.isfinite(sub), np.round(sub, 4), None)
    return {"rows": sub.shape[0], "cols": sub.shape[1],
            "values": [[None if v is None else float(v) for v in row]
                       for row in sub.tolist()]}


# ----------------------------------------------------------------------
# Main exporter
# ----------------------------------------------------------------------

LAYER_SPECS = [
    # (filename, layer id, display name, colormap, vmin, vmax, fmt, unit)
    ("geocore_probability.tif", "prob", "Prospectivity probability",
     "viridis", 0.0, 1.0, ".3f", ""),
    ("geocore_uncertainty.tif", "unc", "Model uncertainty (tree std)",
     "magma", None, None, ".3f", ""),
    ("geocore_classes.tif", "cls", "Target classes (0/1/2)",
     "YlOrRd", 0.0, 2.0, ".0f", ""),
    ("geocore_depth_spectral_m.tif", "dsp", "Source depth - spectral",
     "plasma_r", None, None, ".0f", " m"),
    ("geocore_depth_euler_m.tif", "deu", "Source depth - Euler",
     "plasma_r", None, None, ".0f", " m"),
]


def export_viewer(run_dir: Path | str, out_html: Path | str,
                  deposits_csv: Optional[Path | str] = None,
                  title: str = "GeoCore Analytics - Target Review") -> Path:
    """Build the self-contained HTML viewer from a pipeline run directory."""
    run_dir = Path(run_dir)
    out_html = Path(out_html)

    dem_path = run_dir / "geocore_dem.tif"
    if not dem_path.exists():
        raise FileNotFoundError(f"missing {dem_path}; run the pipeline first")
    dem, transform, crs = _read(dem_path)
    bounds = _bounds_wgs84(dem.shape, transform, crs)

    hs_uri, _, _ = _colormap_png(_hillshade(dem, transform), "gray", 0, 1)

    layers, legends, querygrids = [], {}, {}
    for fname, lid, name, cmap, vmin, vmax, fmt, unit in LAYER_SPECS:
        p = run_dir / fname
        if not p.exists():
            continue
        data, _, _ = _read(p)
        if not np.isfinite(data).any():
            continue
        uri, lo, hi = _colormap_png(data, cmap, vmin, vmax)
        layers.append({"id": lid, "name": name, "png": uri,
                       "fmt": fmt, "unit": unit})
        legends[lid] = {"cmap": cmap, "lo": lo, "hi": hi}
        querygrids[lid] = _value_grid(data)

    deposits = []
    if deposits_csv and Path(deposits_csv).exists():
        import csv as _csv
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        with open(deposits_csv, newline="") as fh:
            rd = _csv.DictReader(fh)
            cols = {c.lower().strip(): c for c in rd.fieldnames or []}
            xc = next((cols[k] for k in
                       ("longitude", "lon", "x", "east", "easting")
                       if k in cols), None)
            yc = next((cols[k] for k in
                       ("latitude", "lat", "y", "north", "northing")
                       if k in cols), None)
            nc = next((cols[k] for k in ("site_name", "name", "dep_name")
                       if k in cols), None)
            if xc and yc:
                for row in rd:
                    try:
                        x = float(row[xc]); y = float(row[yc])
                    except (TypeError, ValueError):
                        continue
                    # Heuristic: values within lat/lon ranges are geographic
                    if abs(x) <= 360 and abs(y) <= 90:
                        lon, lat = x, y
                    else:
                        lon, lat = tr.transform(x, y)
                    deposits.append({"lat": lat, "lon": lon,
                                     "name": (row.get(nc) or "deposit")
                                     if nc else "deposit"})

    # Colorbar gradients for legends
    grads = {}
    for lid, lg in legends.items():
        if _CMAPS is not None:
            ramp = (_CMAPS[lg["cmap"]](np.linspace(0, 1, 64)) * 255
                    ).astype(np.uint8)[None, :, :]
            img = Image.fromarray(ramp, "RGBA").resize((256, 12),
                                                       Image.NEAREST)
        else:
            g = np.tile(np.linspace(0, 255, 256).astype(np.uint8), (12, 1))
            img = Image.fromarray(np.stack([g, g, g], -1), "RGB")
        buf = io.BytesIO(); img.save(buf, format="PNG")
        grads[lid] = ("data:image/png;base64,"
                      + base64.b64encode(buf.getvalue()).decode())

    payload = {
        "title": title, "bounds": bounds, "hillshade": hs_uri,
        "layers": layers, "legends": legends, "gradients": grads,
        "grids": querygrids, "deposits": deposits,
    }

    html = _HTML_TEMPLATE.replace("/*__PAYLOAD__*/",
                                  "const GC = " + json.dumps(payload) + ";")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeoCore Analytics - Target Review</title>
<link rel="stylesheet"
      href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  :root { --bg:#101418; --panel:#1a2129; --text:#e8edf2; --accent:#4cc9f0; }
  html,body { height:100%; margin:0; font-family:system-ui,Segoe UI,Roboto,
              sans-serif; background:var(--bg); color:var(--text); }
  #map { position:absolute; inset:0; }
  #panel { position:absolute; top:12px; right:12px; z-index:1000;
           width:300px; max-height:calc(100% - 24px); overflow:auto;
           background:var(--panel); border-radius:10px; padding:14px 16px;
           box-shadow:0 4px 18px rgba(0,0,0,.5); font-size:13px; }
  #panel h1 { font-size:15px; margin:0 0 4px; color:var(--accent); }
  #panel .sub { color:#9fb0c0; font-size:11px; margin-bottom:10px; }
  .layer { border-top:1px solid #2a3340; padding:8px 0; }
  .layer label { display:flex; align-items:center; gap:8px; cursor:pointer; }
  .layer input[type=range] { width:100%; }
  .legend { display:flex; align-items:center; gap:6px; margin-top:4px;
            font-size:11px; color:#9fb0c0; }
  .legend img { height:10px; flex:1; border-radius:3px; }
  #readout { border-top:1px solid #2a3340; margin-top:8px; padding-top:8px;
             font-size:12px; color:#cfe3f5; min-height:46px; }
  #readout b { color:var(--accent); }
  .note { font-size:10px; color:#7c8b9a; margin-top:10px; line-height:1.5; }
  #offline { position:absolute; inset:0; display:none; place-items:center;
             background:var(--bg); z-index:2000; text-align:center;
             padding:40px; }
</style>
</head>
<body>
<div id="map"></div>
<div id="offline"><div>
  <h2>Map library unavailable</h2>
  <p>This viewer loads Leaflet from a CDN. Connect to the internet once, or
     ask your GeoCore administrator for the offline bundle.</p>
</div></div>
<div id="panel">
  <h1 id="ttl"></h1>
  <div class="sub">Click the map to query values. Toggle layers and adjust
       opacity below.</div>
  <div id="layers"></div>
  <div id="readout">Click the map to read layer values.</div>
  <div class="note">
    Prospectivity values are model probabilities for ranking ground -
    not discovery guarantees. Depth layers estimate magnetic <i>source</i>
    depth below sensor datum, not ore depth. See the validation report for
    metrics and limitations.
  </div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
/*__PAYLOAD__*/
if (typeof L === "undefined") {
  document.getElementById("offline").style.display = "grid";
} else {
  document.getElementById("ttl").textContent = GC.title;
  const map = L.map("map", { zoomSnap: 0.25 });
  map.fitBounds(GC.bounds);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom: 17, attribution: "&copy; OpenStreetMap" }).addTo(map);
  L.imageOverlay(GC.hillshade, GC.bounds, { opacity: 0.85 }).addTo(map);

  const overlays = {}, panel = document.getElementById("layers");
  GC.layers.forEach((ly, i) => {
    const ov = L.imageOverlay(ly.png, GC.bounds, { opacity: 0.7 });
    overlays[ly.id] = { ov, ly, on: i === 0 };
    if (i === 0) ov.addTo(map);
    const lg = GC.legends[ly.id];
    const div = document.createElement("div");
    div.className = "layer";
    div.innerHTML =
      `<label><input type="checkbox" ${i===0?"checked":""} data-id="${ly.id}">
       ${ly.name}</label>
       <input type="range" min="0" max="100" value="70" data-id="${ly.id}">
       <div class="legend"><span>${lg.lo.toFixed(2)}</span>
       <img src="${GC.gradients[ly.id]}">
       <span>${lg.hi.toFixed(2)}${ly.unit}</span></div>`;
    panel.appendChild(div);
  });
  panel.addEventListener("change", e => {
    if (e.target.type !== "checkbox") return;
    const o = overlays[e.target.dataset.id];
    o.on = e.target.checked;
    o.on ? o.ov.addTo(map) : o.ov.remove();
  });
  panel.addEventListener("input", e => {
    if (e.target.type !== "range") return;
    overlays[e.target.dataset.id].ov.setOpacity(e.target.value / 100);
  });

  GC.deposits.forEach(d => L.circleMarker([d.lat, d.lon],
    { radius: 5, color: "#ffd166", weight: 2, fillOpacity: 0.6 })
    .bindTooltip(d.name).addTo(map));

  const [[s, w], [n, eL]] = GC.bounds;
  map.on("click", ev => {
    const { lat, lng } = ev.latlng;
    if (lat < s || lat > n || lng < w || lng > eL) return;
    const fy = (n - lat) / (n - s), fx = (lng - w) / (eL - w);
    let out = `<b>${lat.toFixed(5)}, ${lng.toFixed(5)}</b><br>`;
    GC.layers.forEach(ly => {
      const g = GC.grids[ly.id];
      const r = Math.min(g.rows - 1, Math.floor(fy * g.rows));
      const c = Math.min(g.cols - 1, Math.floor(fx * g.cols));
      const v = g.values[r][c];
      out += `${ly.name}: <b>${v === null ? "no data"
              : v.toFixed(ly.fmt === ".0f" ? 0 : 3) + ly.unit}</b><br>`;
    });
    document.getElementById("readout").innerHTML = out;
  });
}
</script>
</body>
</html>
"""
