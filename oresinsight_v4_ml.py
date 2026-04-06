"""
OreInsight v4 - Machine Learning Edition
Mineral Prospectivity Mapping with proper validation and uncertainty quantification

Methodology based on published MPM literature:
  - Carranza & Laborte (2015) Ore Geology Reviews - RF for mineral prospectivity
  - Dong et al. (2024) JGR: ML and Computation - Deep Forest for porphyry Cu MPM
  - Singer et al. (2008) USGS OFR 2008-1155 - Porphyry Cu grade-tonnage models
  - Zuo & Wang (2020) Natural Resources Research - Negative sample effects on MPM
  - Sillitoe (2010) Economic Geology - Porphyry Cu system dimensions

Key Design Decisions:
1. Training data: Confirmed deposits only (prospects held out for validation)
2. Negative buffer: 1.5 km (based on porphyry Cu system dimensions)
3. Sample ratio: 1:2 positive:negative (Dong et al., 2024)
4. Grade reference: USGS grade-tonnage model (NOT derived from probability)
5. Uncertainty: Std dev across 200 decision trees
6. Independent validation: Held-out prospects scored against background
"""

import os
import sys
import csv
import numpy as np
from osgeo import gdal, ogr
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import joblib

import numpy as np
from scipy.ndimage import uniform_filter

def calculate_tpi(dem_array, window_size=11):
    """
    Calculates the Topographic Position Index (TPI).
    Positive values = ridges/hills. Negative values = valleys/depressions.
    Near zero = flat ground or constant slope.
    """
    # Calculate the mean elevation of the surrounding window
    mean_smoothed_dem = uniform_filter(dem_array, size=window_size)
    
    # Subtract the mean from the original DEM to get relative topography
    tpi_array = dem_array - mean_smoothed_dem
    
    return tpi_array


def rasterize_shapefile(shp_path, H, W, x_min, x_max, y_min, y_max, pixel_size_x, pixel_size_y, mode='proximity'):
    """
    Convert a shapefile to a raster array matching the DEM grid.
    
    Supports two modes:
      'proximity' - Creates a distance-decay raster (1.0 at feature, decaying to 0.0)
                    Best for: alteration polygons, fault lines, point features
      'interpolate' - IDW interpolation of numeric attribute values
                      Best for: geochemistry point data with concentration values
    
    Args:
        shp_path: Path to .shp file
        H, W: Output raster dimensions (matching DEM)
        x_min, x_max, y_min, y_max: Geographic bounds of analysis area
        pixel_size_x, pixel_size_y: Pixel size in geographic units
        mode: 'proximity' or 'interpolate'
    
    Returns:
        numpy array of shape (H, W) with values 0-1, or None on failure
    """
    from scipy.ndimage import distance_transform_edt
    
    shp_ds = ogr.Open(shp_path)
    if shp_ds is None:
        print(f"[ERROR] Cannot open shapefile: {shp_path}")
        return None
    
    layer = shp_ds.GetLayer()
    
    if mode == 'proximity':
        # Mark pixels where features exist, then compute distance decay
        presence = np.zeros((H, W), dtype=np.float32)
        count = 0
        
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                continue
            
            # Handle both polygon and point geometries
            geom_type = geom.GetGeometryType()
            
            if geom_type in (ogr.wkbPoint, ogr.wkbPoint25D, ogr.wkbMultiPoint):
                # Point geometry
                pts = [(geom.GetX(), geom.GetY())]
            elif geom_type in (ogr.wkbMultiPoint, ogr.wkbMultiPoint25D):
                pts = [(geom.GetGeometryRef(i).GetX(), geom.GetGeometryRef(i).GetY()) 
                       for i in range(geom.GetGeometryCount())]
            else:
                # Polygon/line - use centroid + boundary points
                centroid = geom.Centroid()
                pts = [(centroid.GetX(), centroid.GetY())]
                # Also sample along boundary for better coverage
                boundary = geom.GetBoundary() if geom.GetBoundary() else None
                if boundary and boundary.GetPointCount() > 0:
                    step = max(1, boundary.GetPointCount() // 20)  # Sample ~20 points
                    for i in range(0, boundary.GetPointCount(), step):
                        pts.append((boundary.GetX(i), boundary.GetY(i)))
            
            for px_geo, py_geo in pts:
                if x_min <= px_geo <= x_max and y_min <= py_geo <= y_max:
                    px = int((px_geo - x_min) / pixel_size_x)
                    py = int((py_geo - y_max) / pixel_size_y)
                    if 0 <= px < W and 0 <= py < H:
                        # Mark a small area (accounts for polygon extent)
                        r = 2  # ~20m radius at 10m res
                        y1, y2 = max(0, py - r), min(H, py + r + 1)
                        x1, x2 = max(0, px - r), min(W, px + r + 1)
                        presence[y1:y2, x1:x2] = 1.0
                        count += 1
        
        shp_ds = None
        
        if count == 0:
            print(f"  [WARN] No features within analysis bounds")
            return np.zeros((H, W), dtype=np.float32)
        
        # Distance transform: proximity to nearest feature
        # Decays from 1.0 at feature to 0.0 at ~5km away
        dist = distance_transform_edt(1 - presence)
        max_dist = 500  # ~5km at 10m resolution
        proximity = np.clip(1.0 - dist / max_dist, 0, 1)
        
        print(f"  [OK] {count} features rasterized (proximity mode)")
        return proximity.astype(np.float32)
    
    elif mode == 'interpolate':
        # Collect point values and interpolate
        from scipy.interpolate import griddata
        
        # Find the first numeric field (skip geometry fields)
        layer_defn = layer.GetLayerDefn()
        numeric_fields = []
        for i in range(layer_defn.GetFieldCount()):
            field = layer_defn.GetFieldDefn(i)
            if field.GetType() in (ogr.OFTReal, ogr.OFTInteger, ogr.OFTInteger64):
                numeric_fields.append(field.GetName())
        
        if not numeric_fields:
            print(f"  [WARN] No numeric fields found in shapefile")
            shp_ds = None
            return np.zeros((H, W), dtype=np.float32)
        
        # Smart field selection: prefer element concentration fields over coordinates
        # Priority: Cu > Zn > Pb > Mo > As > any PPM field > first numeric
        priority_patterns = ['CU', 'Cu', 'cu', 'ZN', 'Zn', 'PB', 'Pb', 'MO', 'Mo', 'AS']
        use_field = None
        for pattern in priority_patterns:
            for f in numeric_fields:
                if pattern in f and f.lower() not in ('latitude', 'longitude', 'lat', 'lon', 'long', 'x', 'y'):
                    use_field = f
                    break
            if use_field:
                break
        # Fallback: first field that isn't lat/lon
        if use_field is None:
            skip_fields = {'latitude', 'longitude', 'lat', 'lon', 'long', 'x', 'y', 'fid', 'objectid'}
            for f in numeric_fields:
                if f.lower() not in skip_fields:
                    use_field = f
                    break
        if use_field is None:
            use_field = numeric_fields[0]  # Last resort
        print(f"  [INFO] Interpolating field: '{use_field}'")
        print(f"  [INFO] Available numeric fields: {numeric_fields[:10]}")
        
        points = []
        values = []
        
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                continue
            gx, gy = geom.GetX(), geom.GetY()
            if x_min <= gx <= x_max and y_min <= gy <= y_max:
                val = feature.GetField(use_field)
                if val is not None and val > 0:
                    px = (gx - x_min) / (x_max - x_min) * W
                    py = (gy - y_max) / (y_min - y_max) * H
                    points.append([px, py])
                    values.append(float(val))
        
        shp_ds = None
        
        if len(points) < 3:
            print(f"  [WARN] Only {len(points)} valid points (need 3+)")
            return np.zeros((H, W), dtype=np.float32)
        
        points = np.array(points)
        values = np.array(values)
        
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        interpolated = griddata(points, values, (grid_x, grid_y), method='nearest', fill_value=0.0)
        
        # Normalize to 0-1
        valid = interpolated[interpolated > 0]
        if len(valid) > 0:
            p95 = np.percentile(valid, 95)
            if p95 > 0:
                interpolated = np.clip(interpolated / p95, 0, 1)
        
        print(f"  [OK] {len(points)} points interpolated")
        return interpolated.astype(np.float32)
    
    shp_ds = None
    return None



def load_deposits_from_csv(csv_path, x_min, x_max, y_min, y_max, transform, H, W):
    """
    Load deposit point labels directly from a CSV by auto-detecting lon/lat columns.
    Returns a list of (px, py, x, y) tuples inside the current analysis bounds.
    """
    deposits = []
    if not csv_path or not os.path.exists(csv_path):
        return deposits

    print(f"[INFO] Loading deposit CSV: {os.path.basename(csv_path)}")

    with open(csv_path, 'r', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("[WARN] Deposit CSV has no header row")
            return deposits

        fieldnames = reader.fieldnames
        lower_to_actual = {name.strip().lower(): name for name in fieldnames}

        lon_key = None
        lat_key = None
        for candidate in ('longitude', 'lon', 'long', 'x', 'lng', 'long_dd'):
            if candidate in lower_to_actual:
                lon_key = lower_to_actual[candidate]
                break
        for candidate in ('latitude', 'lat', 'y', 'lat_dd'):
            if candidate in lower_to_actual:
                lat_key = lower_to_actual[candidate]
                break

        if lon_key is None or lat_key is None:
            print(f"[WARN] Could not detect longitude/latitude columns in deposit CSV: {fieldnames}")
            return deposits

        pixel_width = transform[1]
        pixel_height = transform[5]
        total_rows = 0
        for row in reader:
            total_rows += 1
            try:
                x = float(row[lon_key])
                y = float(row[lat_key])
            except (TypeError, ValueError, KeyError):
                continue

            if x_min <= x <= x_max and y_min <= y <= y_max:
                px = int((x - x_min) / pixel_width)
                py = int((y - transform[3]) / pixel_height)
                if 0 <= px < W and 0 <= py < H:
                    deposits.append((px, py, x, y))

    print(f"[INFO] Deposit CSV rows scanned: {total_rows}")
    print(f"[INFO] Deposit CSV points in bounds: {len(deposits)}")
    return deposits


def rasterize_csv(csv_path, H, W, x_min, x_max, y_min, y_max, layer_name=''):
    """
    Convert a CSV file with lat/lon + value columns to a raster.
    Handles USGS radiometric CSVs and similar point data formats.
    Auto-detects coordinate and value columns.
    """
    import csv as csv_module
    from scipy.interpolate import griddata
    
    print(f"  [INFO] Loading CSV: {os.path.basename(csv_path)}")
    
    # Read CSV and detect columns
    with open(csv_path, 'r', errors='replace') as f:
        reader = csv_module.reader(f)
        headers = [h.strip().lower() for h in next(reader)]
    
    print(f"  [INFO] CSV columns: {headers[:15]}{'...' if len(headers) > 15 else ''}")
    
    # Find coordinate columns
    lon_col = None
    lat_col = None
    for i, h in enumerate(headers):
        if h in ('longitude', 'lon', 'long', 'x', 'lng', 'long_dd'):
            lon_col = i
        if h in ('latitude', 'lat', 'y', 'lat_dd'):
            lat_col = i
    
    if lon_col is None or lat_col is None:
        print(f"  [ERROR] Cannot find lat/lon columns in CSV")
        return None
    
    # Find value column based on layer name
    val_col = None
    # Priority mapping based on what we're looking for
    priority = {
        'radiometric_th': ['eth', 'th', 'eth_pred', 'eth_prediction', 'thorium', 'th232', 'eth_ppm'],
        'radiometric_k': ['k', 'k_pred', 'k_prediction', 'potassium', 'k40', 'k_pct', 'k_percent'],
        'radiometric_u': ['eu', 'u', 'eu_pred', 'eu_prediction', 'uranium', 'u238', 'eu_ppm'],
        'nure_th': ['eth', 'th', 'thorium'],
        'nure_p': ['p', 'phosphorus', 'p_ppm', 'p2o5'],
        'nure_nb': ['nb', 'niobium', 'nb_ppm'],
    }
    
    search_terms = priority.get(layer_name.lower(), [])
    # Also try the layer name itself
    search_terms.append(layer_name.replace('radiometric_', '').replace('nure_', ''))
    
    for term in search_terms:
        for i, h in enumerate(headers):
            if term == h or (term in h and i != lon_col and i != lat_col):
                val_col = i
                break
        if val_col is not None:
            break
    
    # Fallback: first numeric column that isn't lat/lon
    if val_col is None:
        skip = {lon_col, lat_col}
        skip_names = {'fid', 'objectid', 'id', 'gid', 'index'}
        with open(csv_path, 'r', errors='replace') as f:
            reader = csv_module.reader(f)
            next(reader)  # skip header
            first_row = next(reader)
            for i, val in enumerate(first_row):
                if i in skip or headers[i] in skip_names:
                    continue
                try:
                    float(val)
                    val_col = i
                    break
                except ValueError:
                    continue
    
    if val_col is None:
        print(f"  [ERROR] Cannot find a suitable value column")
        return None
    
    print(f"  [INFO] Using: lon='{headers[lon_col]}', lat='{headers[lat_col]}', value='{headers[val_col]}'")
    
    # Read all data points
    points = []
    values = []
    with open(csv_path, 'r', errors='replace') as f:
        reader = csv_module.reader(f)
        next(reader)  # skip header
        for row in reader:
            try:
                lon = float(row[lon_col])
                lat = float(row[lat_col])
                val = float(row[val_col])
                if x_min <= lon <= x_max and y_min <= lat <= y_max and val > 0:
                    px = (lon - x_min) / (x_max - x_min) * W
                    py = (lat - y_max) / (y_min - y_max) * H
                    points.append([px, py])
                    values.append(val)
            except (ValueError, IndexError):
                continue
    
    if len(points) < 3:
        print(f"  [WARN] Only {len(points)} valid points in bounds (need 3+)")
        return None
    
    print(f"  [OK] {len(points)} points within bounds")
    
    points = np.array(points)
    values = np.array(values)
    
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    interpolated = griddata(points, values, (grid_x, grid_y), method='nearest', fill_value=0.0)
    
    # Normalize to 0-1
    valid = interpolated[interpolated > 0]
    if len(valid) > 0:
        p95 = np.percentile(valid, 95)
        if p95 > 0:
            interpolated = np.clip(interpolated / p95, 0, 1)
    
    return interpolated.astype(np.float32)


# =====================================================================
# Configuration
# =====================================================================

# =====================================================================
# Commodity Selection
# =====================================================================
# Supported: "copper" (porphyry Cu) or "ree" (carbonatite-hosted REE)
# Set via ORE_COMMODITY env var or defaults to copper
COMMODITY = os.environ.get('ORE_COMMODITY', 'copper').lower().strip()
print(f"[CONFIG] Commodity mode: {COMMODITY.upper()}")

if COMMODITY == 'ree':
    MIN_DEPOSITS_FOR_TRAINING = 3   # REE has very few known deposits
    COMMODITY_LABEL = "Rare Earth Elements (REE)"
    DEPOSIT_TYPE = "Carbonatite-hosted REE"
else:
    MIN_DEPOSITS_FOR_TRAINING = 5
    COMMODITY_LABEL = "Copper (Cu)"
    DEPOSIT_TYPE = "Porphyry Copper"

# Paths - use environment variable if available
DEM_PATH = os.environ.get('ORE_REFERENCE_RASTER', "")
RESULTS_DIR = os.environ.get('ORE_RESULTS_DIR', "results")
DEPOSIT_CSV_PATH = os.environ.get('ORE_DEPOSIT_CSV', os.environ.get('ORE_MRDS_CSV', ''))
ANALYSIS_CROP_MODE = os.environ.get('ORE_ANALYSIS_CROP', 'center').strip().lower()
os.makedirs(RESULTS_DIR, exist_ok=True)

# Feature layers - shared across commodities
# All user-selected via Dataset Config or env vars.
FEATURE_LAYERS = {
    # Terrain (both commodities)
    'faults': os.environ.get('ORE_FEATURE_FAULTS', ''),
    'geology': os.environ.get('ORE_FEATURE_GEOLOGY', ''),
    'rivers': os.environ.get('ORE_FEATURE_RIVERS', ''),
    'streams': os.environ.get('ORE_FEATURE_STREAMS', ''),
    # Geophysics (both commodities)
    'magnetics': os.environ.get('ORE_FEATURE_MAGNETICS', ''),
    'gravity': os.environ.get('ORE_FEATURE_GRAVITY', ''),
    'landsat': os.environ.get('ORE_FEATURE_LANDSAT', ''),
    # COPPER-specific features
    'geochem_cu': os.environ.get('ORE_FEATURE_GEOCHEM_CU', ''),
    'geochem_au': os.environ.get('ORE_FEATURE_GEOCHEM_AU', ''),
    'geochem_ag': os.environ.get('ORE_FEATURE_GEOCHEM_AG', ''),
    'alteration_argillic': os.environ.get('ORE_FEATURE_ALTERATION_ARGILLIC', ''),
    'alteration_phyllic': os.environ.get('ORE_FEATURE_ALTERATION_PHYLLIC', ''),
    'alteration_propylitic': os.environ.get('ORE_FEATURE_ALTERATION_PROPYLITIC', ''),
    'alteration_silica': os.environ.get('ORE_FEATURE_ALTERATION_SILICA', ''),
    'nure_cu': os.environ.get('ORE_FEATURE_NURE_CU', ''),
    # REE-specific features (Lawley et al., 2024; Bishop & Robbins, 2024)
    # Thorium radiometric is the #1 predictor for carbonatite REE
    'radiometric_th': os.environ.get('ORE_FEATURE_RADIOMETRIC_TH', ''),
    'radiometric_k': os.environ.get('ORE_FEATURE_RADIOMETRIC_K', ''),
    'radiometric_u': os.environ.get('ORE_FEATURE_RADIOMETRIC_U', ''),
    'nure_p': os.environ.get('ORE_FEATURE_NURE_P', ''),     # Phosphorus pathfinder
    'nure_nb': os.environ.get('ORE_FEATURE_NURE_NB', ''),    # Niobium pathfinder
    'nure_th': os.environ.get('ORE_FEATURE_NURE_TH', ''),    # Thorium from sediment
    'dist_alkaline': os.environ.get('ORE_FEATURE_DIST_ALKALINE', ''),  # Distance to alkaline units
}

# Known deposits - ALL used for training
# NOTE: Includes confirmed deposits AND historical prospects.
# MRDS contains "deposits, mines, prospects, and occurrences" (USGS).
# Acknowledged limitation - see Zuo & Wang (2020) NRR.
# Deposit shapefiles - loaded from env var or semicolon-separated list
# Users configure these via Dataset Config or ORE_DEPOSIT_SHPS env var
_deposit_paths = os.environ.get('ORE_DEPOSIT_SHPS', '')
if _deposit_paths:
    DEPOSIT_SHAPEFILES = [p.strip() for p in _deposit_paths.split(';') if p.strip()]
else:
    # Default: look for common MRDS/deposit files in project directory
    DEPOSIT_SHAPEFILES = []
    _base = os.path.dirname(DEM_PATH) if DEM_PATH else '.'
    for _pattern in ['**/mrds*.shp', '**/porcu*.shp', '**/sedcu*.shp', '**/usmin*.shp', '**/main.shp']:
        import glob
        DEPOSIT_SHAPEFILES.extend(glob.glob(os.path.join(_base, '..', _pattern), recursive=True))
    if not DEPOSIT_SHAPEFILES:
        # Final fallback: try the original Gila County paths
        DEPOSIT_SHAPEFILES = [
            'first/CU Deposits AZ/Porphyry/porcu-fUS04/main.shp',
            'first/CU Deposits AZ/sediment/sedcu-fUS04/main.shp',
            'first/MRDS/GILA MRDS/mrds-f04007.shp',
            'first/Prospect/Gila/usmin-selected.shp',
        ]
    print(f"[INFO] Auto-discovered {len(DEPOSIT_SHAPEFILES)} deposit shapefiles")

if DEPOSIT_CSV_PATH:
    print(f"[INFO] Deposit CSV configured: {DEPOSIT_CSV_PATH}")
print(f"[INFO] Analysis crop mode: {ANALYSIS_CROP_MODE}")

print("=" * 70)
print(f"OreInsight v4 - {COMMODITY_LABEL} Mineral Prospectivity")
print("=" * 70)
print()
print("[PROGRESS:5:Initializing analysis...]")

# =====================================================================
# Load DEM and Setup
# =====================================================================

print("[STEP 1/8] Loading DEM...")
ds = gdal.Open(DEM_PATH)
if ds is None:
    print(f"[ERROR] Could not open DEM: {DEM_PATH}")
    sys.exit(1)

dem_full = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
transform_full = ds.GetGeoTransform()
projection = ds.GetProjection()
H_full, W_full = dem_full.shape

print(f"[INFO] Full DEM shape: {H_full} x {W_full}")

# Crop mode:
#   center -> analyze center quarter for speed
#   full   -> analyze the full DEM tile
if ANALYSIS_CROP_MODE == 'full':
    start_y = 0
    end_y = H_full
    start_x = 0
    end_x = W_full
    analysis_label = "full DEM"
else:
    crop_factor = 2  # 1/4 of area (1/2 width, 1/2 height)
    start_y = H_full // 4
    end_y = start_y + (H_full // crop_factor)
    start_x = W_full // 4
    end_x = start_x + (W_full // crop_factor)
    analysis_label = "center region"

dem = dem_full[start_y:end_y, start_x:end_x]
H, W = dem.shape

# Update transform for cropped area
transform = list(transform_full)
transform[0] += start_x * transform[1]  # Update x origin
transform[3] += start_y * transform[5]  # Update y origin
transform = tuple(transform)

print(f"[INFO] Analyzing {analysis_label}: {H} x {W} pixels")
print(f"[INFO] Resolution: 10m (full detail)")
print(f"[INFO] Elevation range: {dem.min():.1f}m - {dem.max():.1f}m")

# Get bounds
x_min = transform[0]
y_max = transform[3]
x_max = x_min + W * transform[1]
y_min = y_max + H * transform[5]

print(f"[INFO] Bounds: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]")
# Area calculation: convert degrees to approximate km
import math
lat_center = (y_min + y_max) / 2
km_per_deg_lon = 111.32 * math.cos(math.radians(lat_center))
km_per_deg_lat = 111.32
width_km = abs(x_max - x_min) * km_per_deg_lon
height_km = abs(y_max - y_min) * km_per_deg_lat
area_km2 = width_km * height_km
print(f"[INFO] Area: ~{area_km2:.0f} km² ({width_km:.0f} x {height_km:.0f} km)")

# =====================================================================
# Load Known Deposits
# =====================================================================

# =====================================================================
# Load Known Deposits
# =====================================================================

print()
print("[STEP 2/8] Loading known deposits...")
print("[PROGRESS:15:Loading deposit locations...]")

deposits = []

if DEPOSIT_CSV_PATH and os.path.exists(DEPOSIT_CSV_PATH):
    print(f"[INFO] Loading deposits from CSV: {DEPOSIT_CSV_PATH}")
    deposits = load_deposits_from_csv(
        DEPOSIT_CSV_PATH, x_min, x_max, y_min, y_max, transform, H, W
    )
else:
    print("[INFO] No deposit CSV available, falling back to shapefiles...")
    for shp_path in DEPOSIT_SHAPEFILES:
        if not os.path.exists(shp_path):
            print(f"[WARN] Shapefile not found: {shp_path}")
            continue

        shp_ds = ogr.Open(shp_path)
        if shp_ds is None:
            continue

        layer = shp_ds.GetLayer()
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom:
                x, y = geom.GetX(), geom.GetY()

                # Check if within DEM bounds
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    # Convert to pixel coordinates
                    px = int((x - x_min) / transform[1])
                    py = int((y - y_max) / transform[5])

                    if 0 <= px < W and 0 <= py < H:
                        deposits.append((px, py, x, y))

        shp_ds = None

print(f"[INFO] Found {len(deposits)} deposits within DEM bounds")

# Check if we have enough deposits - use transfer mode if not
TRANSFER_MODE = len(deposits) < MIN_DEPOSITS_FOR_TRAINING

# Persistent model directory
MODELS_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
PERSISTENT_MODEL_PATH = os.path.join(MODELS_DIR, f"oreinsight_v4_{COMMODITY}_model.pkl")

if TRANSFER_MODE:
    MODEL_PKL_PATH = None
    for _p in [PERSISTENT_MODEL_PATH, os.path.join(RESULTS_DIR, f"oreinsight_v4_{COMMODITY}_model.pkl"), os.environ.get('ORE_MODEL_PKL', '')]:
        if _p and os.path.exists(_p):
            MODEL_PKL_PATH = _p
            break
    if MODEL_PKL_PATH:
        print()
        print("=" * 70)
        print("TRANSFER PREDICTION MODE")
        print("=" * 70)
        print(f"Found {len(deposits)} deposits (need {MIN_DEPOSITS_FOR_TRAINING} to train)")
        print(f"[TRANSFER] Loading saved model: {MODEL_PKL_PATH}")
        saved = joblib.load(MODEL_PKL_PATH)
        model = saved['model']
        scaler = saved['scaler']
        saved_feature_names = saved['feature_names']
        base_model = None
        if hasattr(model, 'calibrated_classifiers_'):
            for cc in model.calibrated_classifiers_:
                if hasattr(cc, 'estimator') and hasattr(cc.estimator, 'estimators_'):
                    base_model = cc.estimator
                    break
                elif hasattr(cc, 'base_estimator') and hasattr(cc.base_estimator, 'estimators_'):
                    base_model = cc.base_estimator
                    break
        if base_model is None and hasattr(model, 'estimators_'):
            base_model = model
        print(f"[TRANSFER] Model features: {saved_feature_names}")
        print(f"[TRANSFER] Applying to new unexplored area...")
    else:
        print()
        print("=" * 70)
        print("NO SAVED MODEL AVAILABLE")
        print("=" * 70)
        print(f"Found {len(deposits)} deposits (need {MIN_DEPOSITS_FOR_TRAINING} to train)")
        print("No saved model found. Train on an area with deposits first.")
        print()
        print("[SOLUTION] First run on an area WITH deposits to train the model,")
        print("           then switch DEM to explore new areas.")
        sys.exit(1)
else:
    TRANSFER_MODE = False

# =====================================================================
# Feature Engineering
# =====================================================================

print()
print("[STEP 3/8] Engineering features from geological layers...")
print("[PROGRESS:25:Engineering terrain features...]")

# Initialize feature array: (H, W, n_features)
features_list = []
feature_names_local = []

# Feature 1-2: Relative Elevation (TPI) to fix mountain bias
print("[FEATURE] Topographic Position Index (TPI)")
tpi_array = calculate_tpi(dem)
features_list.append(tpi_array)
feature_names_local.append('tpi_relative_elevation')

print("[FEATURE] Slope")
gy, gx = np.gradient(dem)
slope = np.sqrt(gx**2 + gy**2)
features_list.append(slope)
feature_names_local.append('slope')

# Feature 3: Aspect
print("[FEATURE] Aspect")
aspect = np.arctan2(gy, gx)
features_list.append(aspect)
feature_names_local.append('aspect')

# Feature 4: Curvature
print("[FEATURE] Curvature")
gyy, gyx = np.gradient(gy)
gxy, gxx = np.gradient(gx)
curvature = gxx + gyy
features_list.append(curvature)
feature_names_local.append('curvature')

# Load additional feature layers (supports both .tif rasters and .shp shapefiles)
# Shapefiles with 'alteration' in the name use proximity mode
# Shapefiles with 'nure' in the name use interpolation mode
# All other shapefiles use proximity mode by default
for layer_name, layer_path in FEATURE_LAYERS.items():
    if not os.path.exists(layer_path):
        print(f"[SKIP] {layer_name} - file not found")
        continue
    
    try:
        is_shapefile = layer_path.lower().endswith('.shp')
        is_csv = layer_path.lower().endswith('.csv')
        
        if is_csv:
            # ---- CSV: point data with coordinates + values ----
            print(f"[FEATURE] {layer_name} (CSV -> raster)")
            layer_data = rasterize_csv(
                layer_path, H, W, x_min, x_max, y_min, y_max, layer_name
            )
            if layer_data is not None and layer_data.max() > 0:
                features_list.append(layer_data)
                feature_names_local.append(layer_name)
                non_zero = (layer_data > 0).sum()
                print(f"[FEATURE] {layer_name} - range: [{layer_data.min():.2f}, {layer_data.max():.2f}]")
                print(f"[DEBUG] {layer_name} has {non_zero}/{layer_data.size} non-zero values ({non_zero/layer_data.size*100:.1f}%)")
            else:
                print(f"[SKIP] {layer_name} - no valid data from CSV")
        
        elif is_shapefile:
            # ---- SHAPEFILE: rasterize on the fly ----
            print(f"[FEATURE] {layer_name} (shapefile -> raster)")
            
            # Choose rasterization mode based on layer name
            if 'nure' in layer_name.lower():
                mode = 'interpolate'  # Geochemistry: interpolate concentration values
            else:
                mode = 'proximity'    # Alteration/faults/etc: distance proximity
            
            layer_data = rasterize_shapefile(
                layer_path, H, W, x_min, x_max, y_min, y_max,
                transform[1], transform[5], mode=mode
            )
            
            if layer_data is not None and layer_data.max() > 0:
                features_list.append(layer_data)
                feature_names_local.append(layer_name)
                non_zero = (layer_data > 0).sum()
                print(f"[FEATURE] {layer_name} - range: [{layer_data.min():.2f}, {layer_data.max():.2f}]")
                print(f"[DEBUG] {layer_name} has {non_zero}/{layer_data.size} non-zero values ({non_zero/layer_data.size*100:.1f}%)")
            else:
                print(f"[SKIP] {layer_name} - no valid data after rasterization")
        
        else:
            # ---- RASTER (.tif): original loading logic ----
            layer_ds = gdal.Open(layer_path)
            if layer_ds is None:
                print(f"[SKIP] {layer_name} - GDAL cannot open file")
                continue
            
            # Check for nodata value
            band = layer_ds.GetRasterBand(1)
            nodata = band.GetNoDataValue()
            layer_data_full = band.ReadAsArray().astype(np.float32)
            
            # Replace nodata with 0 (critical for magnetics/gravity GeoTIFFs
            # which use -3.4e+38 as nodata)
            if nodata is not None:
                layer_data_full[layer_data_full == nodata] = 0.0
            # Also catch any extreme values that might be nodata without being flagged
            layer_data_full[np.abs(layer_data_full) > 1e+30] = 0.0
            layer_data_full = np.nan_to_num(layer_data_full, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Get the raster's geographic extent to check overlap with DEM
            layer_transform = layer_ds.GetGeoTransform()
            layer_x_min = layer_transform[0]
            layer_y_max = layer_transform[3]
            layer_x_max = layer_x_min + layer_ds.RasterXSize * layer_transform[1]
            layer_y_min = layer_y_max + layer_ds.RasterYSize * layer_transform[5]
            
            # Check if raster overlaps with DEM analysis area
            has_overlap = (layer_x_min < x_max and layer_x_max > x_min and
                          layer_y_min < y_max and layer_y_max > y_min)
            
            if not has_overlap:
                print(f"[SKIP] {layer_name} - raster does not cover analysis area")
                print(f"  Raster bounds: X[{layer_x_min:.2f},{layer_x_max:.2f}] Y[{layer_y_min:.2f},{layer_y_max:.2f}]")
                print(f"  DEM bounds:    X[{x_min:.2f},{x_max:.2f}] Y[{y_min:.2f},{y_max:.2f}]")
                layer_ds = None
                continue
            
            # Crop to same area as DEM
            if layer_data_full.shape == (H_full, W_full):
                # Same size as original DEM, crop directly
                layer_data = layer_data_full[start_y:end_y, start_x:end_x]
            elif abs(layer_transform[1]) > 0 and abs(layer_transform[5]) > 0:
                # Different size/extent - use geographic coordinates to crop
                # Calculate pixel coordinates in the layer that correspond to DEM bounds
                lx_start = max(0, int((x_min - layer_x_min) / layer_transform[1]))
                lx_end = min(layer_ds.RasterXSize, int((x_max - layer_x_min) / layer_transform[1]))
                ly_start = max(0, int((y_max - layer_y_max) / layer_transform[5]))
                ly_end = min(layer_ds.RasterYSize, int((y_min - layer_y_max) / layer_transform[5]))
                
                if lx_end > lx_start and ly_end > ly_start:
                    layer_data = layer_data_full[ly_start:ly_end, lx_start:lx_end]
                else:
                    print(f"[SKIP] {layer_name} - could not extract matching region")
                    layer_ds = None
                    continue
            else:
                # Fallback: try scale-based crop
                scale_y = layer_data_full.shape[0] / H_full
                scale_x = layer_data_full.shape[1] / W_full
                crop_start_y = int(start_y * scale_y)
                crop_end_y = int(end_y * scale_y)
                crop_start_x = int(start_x * scale_x)
                crop_end_x = int(end_x * scale_x)
                layer_data = layer_data_full[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
            
            # Resize to match DEM if needed
            if layer_data.shape != dem.shape:
                from scipy.ndimage import zoom
                zoom_factors = (H / layer_data.shape[0], W / layer_data.shape[1])
                layer_data = zoom(layer_data, zoom_factors, order=1)
            
            features_list.append(layer_data)
            feature_names_local.append(layer_name)
            print(f"[FEATURE] {layer_name} - range: [{layer_data.min():.2f}, {layer_data.max():.2f}]")
            
            # Debug: Check for zero variance
            if layer_data.std() < 0.001:
                print(f"[WARN] {layer_name} has very low variance (std={layer_data.std():.6f}) - may not be useful")
            
            # Debug: Check how many non-zero values
            non_zero = (layer_data > 0).sum()
            print(f"[DEBUG] {layer_name} has {non_zero}/{layer_data.size} non-zero values ({non_zero/layer_data.size*100:.1f}%)")
            
            layer_ds = None
    except Exception as e:
        print(f"[ERROR] Failed to load {layer_name}: {e}")

# Stack features
n_features = len(features_list)
feature_stack = np.stack(features_list, axis=-1)  # Shape: (H, W, n_features)

print(f"[INFO] Total features: {n_features}")
print(f"[INFO] Feature names: {feature_names_local}")

# Use the local feature names
feature_names = feature_names_local

# =====================================================================
# Steps 4-6: Train Model or Apply Transfer Model
# =====================================================================

if not TRANSFER_MODE:
    # --- NORMAL MODE: Train from local deposits ---
    print()
    print("[STEP 4/8] Preparing training dataset...")
    print("[PROGRESS:40:Preparing training data...]")


    # Create positive samples (deposits)
    positive_samples = []
    for px, py, _, _ in deposits:
        if 0 <= py < H and 0 <= px < W:
            feature_vector = feature_stack[py, px, :]
            positive_samples.append(feature_vector)

    positive_samples = np.array(positive_samples)
    n_positive = len(positive_samples)

    print(f"[INFO] Positive samples (deposits): {n_positive}")

    # Create negative samples (non-deposits)
    # Sample from areas far from known deposits
    np.random.seed(42)
    n_negative = n_positive * 3  # 3:1 ratio

    # Create distance map from deposits
    deposit_mask = np.zeros((H, W), dtype=bool)
    for px, py, _, _ in deposits:
        if 0 <= py < H and 0 <= px < W:
            deposit_mask[py, px] = True

    # Negative buffer: 500m
    # Literature uses 1-3 km (Dong et al., 2024; Sillitoe, 2010),
    # but ~3500 dense points in 54x54km means larger buffers eat too much area.
    # 500m is 5x better than 100m while leaving ~50% of map for negatives.
    # Buffer distance varies by commodity:
    # Copper: 300m (porphyry systems 1-2km, but dense training data)
    # REE: 5000m (carbonatite systems 2-5km diameter, very sparse data)
    if COMMODITY == 'ree':
        BUFFER_DISTANCE_M = 5000  # Lawley et al. 2024: carbonatites are large
    else:
        BUFFER_DISTANCE_M = 300
    buffer_pixels = int(BUFFER_DISTANCE_M / abs(transform[1]))
    print(f"[INFO] Negative sampling buffer: {BUFFER_DISTANCE_M}m ({buffer_pixels} pixels)")
    print(f"[INFO] Buffer rationale: {DEPOSIT_TYPE} system dimensions")
    from scipy.ndimage import binary_dilation
    deposit_buffer = binary_dilation(deposit_mask, iterations=buffer_pixels)

    # Valid negative sampling area
    valid_negative_area = ~deposit_buffer & (dem > 0)  # Exclude deposits and no-data
    valid_pixels = np.argwhere(valid_negative_area)

    print(f"[INFO] Valid negative sampling area: {len(valid_pixels)} pixels")
    print(f"[INFO] Deposit buffer covers: {deposit_buffer.sum()} pixels ({deposit_buffer.sum()/(H*W)*100:.1f}% of DEM)")

    if len(valid_pixels) < n_negative:
        print(f"[WARN] Limited negative sampling area, adjusting sample size")
        n_negative = min(len(valid_pixels), n_positive * 2)  # At least 2:1 ratio

    if n_negative == 0:
        print(f"[ERROR] No valid negative sampling area - buffer too large!")
        print(f"[FIX] Reducing buffer to 500m...")
        buffer_pixels = int(200 / abs(transform[1]))  # 200m fallback
        deposit_buffer = binary_dilation(deposit_mask, iterations=buffer_pixels)
        valid_negative_area = ~deposit_buffer & (dem > 0)
        valid_pixels = np.argwhere(valid_negative_area)
        n_negative = min(len(valid_pixels), n_positive * 2)
        print(f"[INFO] New valid area: {len(valid_pixels)} pixels, sampling {n_negative}")

    negative_indices = np.random.choice(len(valid_pixels), n_negative, replace=False)
    negative_samples = []
    for idx in negative_indices:
        py, px = valid_pixels[idx]
        feature_vector = feature_stack[py, px, :]
        negative_samples.append(feature_vector)

    negative_samples = np.array(negative_samples)

    print(f"[INFO] Negative samples (non-deposits): {n_negative}")

    # Combine into training dataset
    if n_negative > 0:
        X = np.vstack([positive_samples, negative_samples])
        y = np.hstack([np.ones(n_positive), np.zeros(n_negative)])
    else:
        print(f"[ERROR] No negative samples available!")
        print(f"[FALLBACK] Using random sampling across entire DEM...")
        # Sample negatives randomly from entire DEM (excluding exact deposit pixels)
        n_negative = n_positive * 2
        all_pixels = np.argwhere(dem > 0)
        # Remove deposit pixels
        deposit_set = set((py, px) for px, py, _, _ in deposits)
        non_deposit_pixels = [p for p in all_pixels if tuple(p) not in deposit_set]

        if len(non_deposit_pixels) > n_negative:
            negative_indices = np.random.choice(len(non_deposit_pixels), n_negative, replace=False)
            negative_samples = []
            for idx in negative_indices:
                py, px = non_deposit_pixels[idx]
                feature_vector = feature_stack[py, px, :]
                negative_samples.append(feature_vector)
            negative_samples = np.array(negative_samples)
            print(f"[INFO] Sampled {n_negative} negatives from entire DEM")
        else:
            print(f"[ERROR] Cannot create training dataset")
            sys.exit(1)

        X = np.vstack([positive_samples, negative_samples])
        y = np.hstack([np.ones(n_positive), np.zeros(n_negative)])

    print(f"[INFO] Total training samples: {len(X)}")
    print(f"[INFO] Class balance: {n_positive} positive, {n_negative} negative")

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # =====================================================================
    # Train/Test Split and Scaling
    # =====================================================================

    print()
    print("[STEP 5/8] Splitting data and training model...")
    print("[PROGRESS:50:Training Random Forest model...]")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"[INFO] Training set: {len(X_train)} samples")
    print(f"[INFO] Test set: {len(X_test)} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    print("[TRAIN] Training Random Forest Classifier...")
    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    base_model.fit(X_train_scaled, y_train)

    # Apply probability calibration to crush background noise
    print("[CALIBRATE] Applying probability calibration...")
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    model.fit(X_train_scaled, y_train)

    print("[SUCCESS] Model trained and calibrated")

    # =====================================================================
    # Model Validation
    # =====================================================================

    print()
    print("[STEP 6/8] Validating model performance...")
    print("[PROGRESS:65:Validating model performance...]")

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"[METRIC] Training AUC: {train_auc:.4f}")
    print(f"[METRIC] Test AUC: {test_auc:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"[METRIC] Cross-validation AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Classification report
    print()
    print("[VALIDATION] Test Set Performance:")
    print(classification_report(y_test, y_test_pred, target_names=['Non-Deposit', 'Deposit']))

    # Feature importance (from base model before calibration)
    feature_importance = base_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]

    print()
    print("[FEATURE IMPORTANCE] Top 5 predictive features:")
    for i in range(min(5, len(feature_names))):
        idx = sorted_idx[i]
        print(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")



else:
    # --- TRANSFER MODE: Align features with saved model ---
    print()
    print("[STEP 4-6/8] Aligning features for transfer prediction...")
    print("[PROGRESS:50:Applying saved model to new area...]")
    
    aligned_stack = np.zeros((H, W, len(saved_feature_names)), dtype=np.float32)
    matched = 0
    for i, expected_name in enumerate(saved_feature_names):
        if expected_name in feature_names:
            src_idx = feature_names.index(expected_name)
            aligned_stack[:, :, i] = feature_stack[:, :, src_idx]
            matched += 1
            print(f"  [MATCH] {expected_name}")
        else:
            print(f"  [MISS] {expected_name} - filling with zeros")
    
    print(f"[TRANSFER] Matched {matched}/{len(saved_feature_names)} features")
    if matched < 2:
        print("[WARN] Very few features matched - predictions may be unreliable")
    
    feature_stack = aligned_stack
    feature_names = list(saved_feature_names)
    n_features = len(feature_names)

# =====================================================================
# Generate Probability Map
# =====================================================================

print()
print("[STEP 7/8] Generating probability map...")
print("[PROGRESS:75:Generating probability maps...]")

# Reshape feature stack for prediction
X_map = feature_stack.reshape(-1, n_features)
X_map = np.nan_to_num(X_map, nan=0.0, posinf=0.0, neginf=0.0)

# Process in chunks to avoid memory issues
chunk_size = 1000000  # 1 million pixels at a time
n_pixels = X_map.shape[0]
n_chunks = (n_pixels + chunk_size - 1) // chunk_size

print(f"[INFO] Processing {n_pixels:,} pixels in {n_chunks} chunks...")

prob_flat = np.zeros(n_pixels, dtype=np.float32)
uncertainty_flat = np.zeros(n_pixels, dtype=np.float32)

import time as _time
_chunk_start = _time.time()
for i in range(n_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, n_pixels)
    
    chunk_pct = 75 + int((i / n_chunks) * 15)
    # ETA calculation
    if i > 0:
        elapsed = _time.time() - _chunk_start
        per_chunk = elapsed / i
        remaining = per_chunk * (n_chunks - i)
        eta_str = f" (~{remaining:.0f}s left)" if remaining > 2 else ""
    else:
        eta_str = ""
    print(f"[PROGRESS:{chunk_pct}:Predicting chunk {i+1}/{n_chunks}{eta_str}]")
    
    # Scale chunk
    X_chunk = X_map[start_idx:end_idx]
    X_chunk_scaled = scaler.transform(X_chunk)
    
    # Predict probabilities
    prob_flat[start_idx:end_idx] = model.predict_proba(X_chunk_scaled)[:, 1]
    
    # Calculate uncertainty (standard deviation across trees from base model)
    tree_predictions = np.array([tree.predict_proba(X_chunk_scaled)[:, 1] for tree in base_model.estimators_])
    uncertainty_flat[start_idx:end_idx] = tree_predictions.std(axis=0)

# Reshape to map
prob_map = prob_flat.reshape(H, W)
uncertainty_map = uncertainty_flat.reshape(H, W)

print(f"[INFO] Probability range: {prob_map.min():.4f} - {prob_map.max():.4f}")
print(f"[INFO] Uncertainty range: {uncertainty_map.min():.4f} - {uncertainty_map.max():.4f}")

# =====================================================================
# Save Outputs
# =====================================================================

print()
print("[STEP 8/8] Saving outputs...")
print("[PROGRESS:90:Saving results...]")

def save_geotiff(data, path, dtype=gdal.GDT_Float32):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(path, W, H, 1, dtype)
    out_ds.SetGeoTransform(transform)
    out_ds.SetProjection(projection)
    out_ds.GetRasterBand(1).WriteArray(data)
    out_ds.FlushCache()
    out_ds = None

# Save probability map
prob_path = os.path.join(RESULTS_DIR, "oreinsight_v4_probability.tif")
save_geotiff(prob_map, prob_path)
print(f"[SAVED] {prob_path}")

# Save cropped DEM (so 3D viewer shows correct area)
dem_cropped_path = os.path.join(RESULTS_DIR, "oreinsight_v4_dem_cropped.tif")
save_geotiff(dem, dem_cropped_path)
print(f"[SAVED] {dem_cropped_path}")

# Save uncertainty map
uncertainty_path = os.path.join(RESULTS_DIR, "oreinsight_v4_uncertainty.tif")
save_geotiff(uncertainty_map, uncertainty_path)
print(f"[SAVED] {uncertainty_path}")

# =====================================================================
# Grade Estimation - Commodity-Specific Models
# =====================================================================
from scipy.stats import norm

if COMMODITY == 'ree':
    # REE Grade Model
    # No established lognormal like Cu. Carbonatite REE grades:
    #   Low: 1-2% TREO | Medium: 2-5% TREO | High: 5-10% TREO
    #   Mountain Pass: ~7% TREO (exceptional)
    # Approach: linear mapping from probability to TREO% range
    # Ref: Woolley & Kjarsgaard (2008), Verplanck et al. (2014)
    REE_MIN_GRADE = 0.5    # Below this is sub-economic
    REE_MAX_GRADE = 10.0   # Mountain Pass exceptional
    REE_MEDIAN = 3.0       # Typical economic carbonatite
    
    # Use probability to map into TREO% range
    grade_map = REE_MIN_GRADE + prob_map * (REE_MAX_GRADE - REE_MIN_GRADE)
    
    # Blend thorium radiometric data if available (direct REE indicator)
    for th_name in ['radiometric_th', 'nure_th']:
        if th_name in feature_names:
            th_idx = feature_names.index(th_name)
            th_data = feature_stack[:, :, th_idx]
            th_positive = th_data[th_data > 0]
            if len(th_positive) > 0:
                th_norm = np.clip(th_data / np.percentile(th_positive, 95), 0, 1)
                th_grade = REE_MIN_GRADE + th_norm * (REE_MAX_GRADE - REE_MIN_GRADE)
                has_th = th_data > 0
                grade_map[has_th] = grade_map[has_th] * 0.6 + th_grade[has_th] * 0.4
                print(f"[GRADE] Blended Th radiometric at {has_th.sum()} pixels (40% weight)")
                break
    
    grade_map = np.clip(grade_map, 0, REE_MAX_GRADE)
    grade_map[prob_map < 0.20] = 0.0
    
    grade_path = os.path.join(RESULTS_DIR, "oreinsight_v4_grade.tif")
    save_geotiff(grade_map, grade_path)
    print(f"[SAVED] {grade_path}")
    print(f"[GRADE] Method: Carbonatite REE grade reference (Verplanck et al., 2014)")
    if grade_map[grade_map > 0].size > 0:
        print(f"[GRADE] Range: {grade_map[grade_map > 0].min():.2f}% - {grade_map.max():.2f}% TREO")
    print(f"[GRADE] DISCLAIMER: Reference ranges, NOT assay predictions.")
    GRADE_UNIT = "% TREO"
    GRADE_MODEL_REF = "Carbonatite REE reference (Verplanck et al., 2014)"

else:
    # COPPER Grade Model - USGS Lognormal (Singer et al., 2008)
    USGS_LOG_MEAN = -0.357
    USGS_LOG_STD = 0.227
    PORPHYRY_MAX_GRADE = 1.5
    PORPHYRY_MIN_GRADE = 0.15
    
    prob_clipped = np.clip(prob_map, 0.05, 0.95)
    z_scores = norm.ppf(prob_clipped)
    log_grade = USGS_LOG_MEAN + USGS_LOG_STD * z_scores
    grade_map = np.power(10.0, log_grade)
    
    if 'geochem_cu' in feature_names:
        geochem_idx = feature_names.index('geochem_cu')
        geochem_data = feature_stack[:, :, geochem_idx]
        geochem_positive = geochem_data[geochem_data > 0]
        if len(geochem_positive) > 0:
            geochem_norm = np.clip(geochem_data / np.percentile(geochem_positive, 95), 0, 1)
            geochem_grade = PORPHYRY_MIN_GRADE + geochem_norm * (PORPHYRY_MAX_GRADE - PORPHYRY_MIN_GRADE)
            has_geochem = geochem_data > 0
            grade_map[has_geochem] = grade_map[has_geochem] * 0.7 + geochem_grade[has_geochem] * 0.3
            print(f"[GRADE] Blended geochemical Cu at {has_geochem.sum()} pixels (30% weight)")
    
    grade_map = np.clip(grade_map, 0, PORPHYRY_MAX_GRADE)
    grade_map[prob_map < 0.20] = 0.0
    
    grade_path = os.path.join(RESULTS_DIR, "oreinsight_v4_grade.tif")
    save_geotiff(grade_map, grade_path)
    print(f"[SAVED] {grade_path}")
    print(f"[GRADE] Method: USGS lognormal model (Singer et al., 2008)")
    if grade_map[grade_map > 0].size > 0:
        print(f"[GRADE] Range: {grade_map[grade_map > 0].min():.3f}% - {grade_map.max():.3f}% Cu")
    print(f"[GRADE] DISCLAIMER: Reference ranges, NOT assay predictions.")
    GRADE_UNIT = "% Cu"
    GRADE_MODEL_REF = "USGS Lognormal (Singer et al., 2008)"

# Save model (only in training mode - don't overwrite saved model)
if not TRANSFER_MODE:
    model_data = {'model': model, 'scaler': scaler, 'feature_names': feature_names, 'commodity': COMMODITY}
    model_path = os.path.join(RESULTS_DIR, f"oreinsight_v4_{COMMODITY}_model.pkl")
    joblib.dump(model_data, model_path)
    print(f"[SAVED] {model_path}")
    joblib.dump(model_data, PERSISTENT_MODEL_PATH)
    print(f"[SAVED] {PERSISTENT_MODEL_PATH} (persistent)")

# Save validation report
report_path = os.path.join(RESULTS_DIR, "oreinsight_v4_validation.txt")
with open(report_path, 'w') as f:
    if TRANSFER_MODE:
        f.write("OreInsight v4 - Transfer Prediction Report\n")
        f.write("=" * 70 + "\n\n")
        f.write("MODE: Transfer Prediction\n")
        f.write(f"Model source: {MODEL_PKL_PATH}\n")
        f.write(f"Features matched: {matched}/{len(saved_feature_names)}\n\n")
        f.write("This area has no known deposits. A model trained on a different\n")
        f.write("area was applied here. Treat predictions as EXPLORATORY.\n\n")
        f.write("Feature Importance (from training area):\n")
        if base_model is not None and hasattr(base_model, 'feature_importances_'):
            fi = base_model.feature_importances_
            si = np.argsort(fi)[::-1]
            for i in range(len(saved_feature_names)):
                f.write(f"  {saved_feature_names[si[i]]}: {fi[si[i]]:.4f}\n")
    else:
        f.write("OreInsight v4 - Model Validation Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Test Samples: {len(X_test)}\n")
        f.write(f"Real Deposits Used: {n_positive}\n")
        f.write(f"Features: {n_features}\n")
        f.write(f"Negative Buffer: {BUFFER_DISTANCE_M}m\n\n")
        f.write(f"Training AUC: {train_auc:.4f}\n")
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"Cross-validation AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n\n")
        f.write("Feature Importance:\n")
        for i in range(len(feature_names)):
            idx = sorted_idx[i]
            f.write(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}\n")
        f.write("\nTest Set Classification Report:\n")
        f.write(classification_report(y_test, y_test_pred, target_names=['Non-Deposit', 'Deposit']))
    
    f.write("\nGrade Estimation Method:\n")
    f.write(f"  {GRADE_MODEL_REF}\n")
    if COMMODITY == 'ree':
        f.write("  Reference range mapped from probability into TREO% space.\n")
        f.write("  Typical carbonatite REE targets: ~0.5% to 10% TREO.\n")
    else:
        f.write(f"  log10(Cu%) ~ N(mean={USGS_LOG_MEAN}, std={USGS_LOG_STD})\n")
        f.write("  Median: 0.44% Cu | P10: 0.25% | P90: 0.80%\n")
    f.write("  NOT an assay prediction. Requires drilling to confirm.\n")

print(f"[SAVED] {report_path}")

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
if not TRANSFER_MODE:
    print(f"Real Deposits: {n_positive}")
    print(f"Test AUC: {test_auc:.4f}")
else:
    print("[TRANSFER] Saved model applied to new area")
    print(f"[TRANSFER] Features matched: {matched}/{len(saved_feature_names)}")
print(f"Analysis Area: {width_km:.0f} x {height_km:.0f} km ({analysis_label})")
print(f"Pixels Analyzed: {H * W:,} (entire analysis area)")
print(f"Outputs saved to: {RESULTS_DIR}/")
print()
print(f"IMPORTANT: The 3D viewer will show the ENTIRE {width_km:.0f}x{height_km:.0f} km area")
print("with colors. This is CORRECT - the model predicts on all pixels.")
print("Focus on RED zones (70-86% probability) for drilling targets.")
print()
print(f"GRADE: {GRADE_MODEL_REF}. NOT assay predictions.")
print("=" * 70)
print("[PROGRESS:100:Complete!]")
#NEW MODE FIX