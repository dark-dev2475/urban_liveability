import sys
import os
import glob
import geopandas as gpd
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
import numpy as np
from scipy.ndimage import generic_filter

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
AOI_FILE = os.path.join(PROJECT_ROOT, 'data', 'aoi', 'aoi.geojson')

# Create processed directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Output files
SPECTRAL_STACK_FILE = os.path.join(PROCESSED_DIR, 'spectral_stack_clipped.tif')
SUPER_STACK_FILE = os.path.join(PROCESSED_DIR, 'super_stack.tif')

# Bands
BANDS_10M = ['B02', 'B03', 'B04', 'B08']
BANDS_20M = ['B05', 'B06', 'B07', 'B11', 'B12', 'B8A']
ALL_BANDS = BANDS_10M + BANDS_20M

# Texture window size
WINDOW_SIZE = 7


# --- Helper Functions ---
def find_safe_folder():
    """Find the first .SAFE folder in the raw data directory."""
    print(f"Searching for .SAFE folder in: {RAW_DIR}")
    safe_folders = glob.glob(os.path.join(RAW_DIR, 'S2*.SAFE'))
    if not safe_folders:
        print("❌ ERROR: No .SAFE folder found in data/raw/")
        sys.exit(1)
    print(f"✅ Found product folder: {os.path.basename(safe_folders[0])}")
    return safe_folders[0]


def find_band_files(safe_folder, bands):
    """Find paths to .jp2 files for given bands."""
    band_paths = {}
    print(f"Locating {len(bands)} band files...")
    for band in bands:
        res = '10m' if band in BANDS_10M else '20m'
        search_pattern = os.path.join(safe_folder, '**', f'*_{band}_{res}.jp2')
        try:
            band_file = glob.glob(search_pattern, recursive=True)[0]
            band_paths[band] = band_file
        except IndexError:
            print(f"❌ ERROR: Could not find band {band} in {safe_folder}")
            sys.exit(1)
    print("✅ All band files located.")
    return band_paths


def peak_to_peak(arr):
    """Compute peak-to-peak contrast."""
    return arr.max() - arr.min()


def clip_to_aoi(dataset, aoi_gdf):
    """Safely clip dataset to AOI geometry."""
    if aoi_gdf.crs != dataset.rio.crs:
        aoi_gdf = aoi_gdf.to_crs(dataset.rio.crs)
    return dataset.rio.clip(aoi_gdf.geometry.values, aoi_gdf.crs, drop=True)


# --- Main Script ---
def main():
    print("--- Stage 3: Preprocessing & Feature Engineering ---")

    # 1. Locate data
    safe_folder = find_safe_folder()
    aoi_gdf = gpd.read_file(AOI_FILE)

    # 2. Find band file paths
    band_paths = find_band_files(safe_folder, ALL_BANDS)

    # --- Part 3A: Spectral Stack ---
    print("\n--- Part 3A: Creating 10-Band Clipped Spectral Stack ---")
    all_clipped_bands = []

    # Use B04 (red, 10m) as reference
    ref_path = band_paths['B04']
    print(f"Using {os.path.basename(ref_path)} as 10m reference...")

    with rxr.open_rasterio(ref_path, masked=True) as ref_ds:
        ref_clipped = clip_to_aoi(ref_ds, aoi_gdf)

        # ✅ Get raster metadata safely
        with rasterio.open(ref_path) as src_ref:
            out_meta = src_ref.profile

        out_meta.update({
            "height": ref_clipped.sizes["y"],
            "width": ref_clipped.sizes["x"],
            "transform": ref_clipped.rio.transform(),
        })

        all_clipped_bands.append(ref_clipped.data[0])
        print("✅ Clipped 10m reference band.")

    # Clip remaining 10m bands
    for band in BANDS_10M:
        if band == 'B04':
            continue
        print(f"Clipping 10m band: {band}...")
        with rxr.open_rasterio(band_paths[band], masked=True) as ds:
            clipped = clip_to_aoi(ds, aoi_gdf)
            all_clipped_bands.append(clipped.data[0])

    # Clip + resample 20m bands to 10m grid
    for band in BANDS_20M:
        print(f"Clipping & resampling 20m band: {band}...")
        with rxr.open_rasterio(band_paths[band], masked=True) as ds_20m:
            ds_resampled = ds_20m.rio.reproject_match(ref_clipped, resampling=Resampling.cubic_spline)
            all_clipped_bands.append(ds_resampled.data[0])

    # Stack and save 10-band result
    spectral_stack = np.stack(all_clipped_bands)
    print(f"✅ Created 10-band spectral stack with shape: {spectral_stack.shape}")

    out_meta['count'] = 10
    out_meta['dtype'] = spectral_stack.dtype

    with rasterio.open(SPECTRAL_STACK_FILE, 'w', **out_meta) as dest:
        dest.write(spectral_stack)
        for i, band_name in enumerate(ALL_BANDS, start=1):
            dest.set_band_description(i, band_name)

    print(f"✅ Saved 10-band spectral stack to: {SPECTRAL_STACK_FILE}")

    # --- Part 3B: Texture Features ---
    print("\n--- Part 3B: Calculating Fast Texture Features ---")

    blue = spectral_stack[0].astype(np.float32)
    green = spectral_stack[1].astype(np.float32)
    red = spectral_stack[2].astype(np.float32)
    nir = spectral_stack[3].astype(np.float32)

    print("  Creating Grayscale band...")
    grayscale = (blue + green + red) / 3.0

    print(f"  Calculating Std. Deviation (7x7) on Grayscale...")
    tex_gray_std = generic_filter(grayscale, np.std, size=WINDOW_SIZE, mode='reflect')

    print(f"  Calculating Contrast (7x7) on Grayscale...")
    tex_gray_ptp = generic_filter(grayscale, peak_to_peak, size=WINDOW_SIZE, mode='reflect')

    print(f"  Calculating Std. Deviation (7x7) on NIR...")
    tex_nir_std = generic_filter(nir, np.std, size=WINDOW_SIZE, mode='reflect')

    print(f"  Calculating Contrast (7x7) on NIR...")
    tex_nir_ptp = generic_filter(nir, peak_to_peak, size=WINDOW_SIZE, mode='reflect')

    print("✅ Texture features calculated.")

    all_features = [
        *spectral_stack,
        tex_gray_std,
        tex_gray_ptp,
        tex_nir_std,
        tex_nir_ptp
    ]

    final_stack = np.stack(all_features).astype(out_meta['dtype'])
    print(f"✅ Created 14-band 'super stack' with shape: {final_stack.shape}")

    out_meta['count'] = 14

    with rasterio.open(SUPER_STACK_FILE, 'w', **out_meta) as dest:
        dest.write(final_stack)
        band_names = ALL_BANDS + ['tex_gray_std', 'tex_gray_ptp', 'tex_nir_std', 'tex_nir_ptp']
        for i, band_name in enumerate(band_names, start=1):
            dest.set_band_description(i, band_name)

    print(f"✅ Saved 14-band super stack to: {SUPER_STACK_FILE}")
    print("\n--- Stage 3 Complete ✅ ---")


if __name__ == "__main__":
    main()
