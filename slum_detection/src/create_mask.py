import sys
import os
import glob
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
TRAINING_DIR = os.path.join(PROJECT_ROOT, 'data', 'training')

# File patterns to search for
SLUM_FILE_PATTERN = 'slums.*'
NON_SLUM_FILE_PATTERN = 'non_slums.*'

# Output file
OUTPUT_MASK = os.path.join(TRAINING_DIR, 'mask.tif')

# Value to burn for "No Data"
# We use 255 because 0 = Non-Slum and 1 = Slum
NO_DATA_VALUE = 255

# --- Helper Functions ---

def find_template_raster():
    """Find a 10m band from the .SAFE folder to use as a template."""
    print(f"Searching for template raster in: {RAW_DIR}")
    # Use B04 (Red band) as the template. Any 10m band will work.
    search_pattern = os.path.join(RAW_DIR, 'S2*.SAFE', '**', '*_B04_10m.jp2')
    template_files = glob.glob(search_pattern, recursive=True)
    
    if not template_files:
        print("❌ ERROR: No 10m band file (*_B04_10m.jp2) found in data/raw/S2*.SAFE/")
        print("This file is needed as a template for alignment.")
        sys.exit(1)
        
    print(f"✅ Found template: {os.path.basename(template_files[0])}")
    return template_files[0]

def find_training_files():
    """Find the slum and non-slum vector files."""
    print(f"Searching for training files in: {TRAINING_DIR}")
    
    try:
        slum_file = glob.glob(os.path.join(TRAINING_DIR, SLUM_FILE_PATTERN))[0]
    except IndexError:
        print(f"❌ ERROR: No slum file found matching '{SLUM_FILE_PATTERN}' in {TRAINING_DIR}")
        sys.exit(1)
        
    try:
        non_slum_file = glob.glob(os.path.join(TRAINING_DIR, NON_SLUM_FILE_PATTERN))[0]
    except IndexError:
        print(f"❌ ERROR: No non-slum file found matching '{NON_SLUM_FILE_PATTERN}' in {TRAINING_DIR}")
        sys.exit(1)
        
    print(f"✅ Found slum file: {os.path.basename(slum_file)}")
    print(f"✅ Found non-slum file: {os.path.basename(non_slum_file)}")
    return slum_file, non_slum_file

def load_and_reproject_vectors(filepath, target_crs):
    """Load a vector file and reproject it to match the raster's CRS."""
    try:
        gdf = gpd.read_file(filepath)
        if gdf.empty:
            print(f"⚠️ Warning: {os.path.basename(filepath)} is empty.")
            return None
        
        # Reproject if CRS does not match
        if gdf.crs != target_crs:
            print(f"  Reprojecting {os.path.basename(filepath)} to {target_crs.to_string()}...")
            gdf = gdf.to_crs(target_crs)
        return gdf
        
    except Exception as e:
        print(f"❌ ERROR: Could not read vector file {filepath}: {e}")
        sys.exit(1)

# --- Main Script ---

def main():
    print("--- Stage 2: Ground Truth Mask Creation (Automated) ---")
    
    # 1. Find the template raster to get CRS, transform, and shape
    template_raster_path = find_template_raster()
    
    with rasterio.open(template_raster_path) as src:
        target_crs = src.crs
        transform = src.transform
        out_shape = src.shape
        out_meta = src.meta.copy()
        print(f"Template raster CRS is: {target_crs.to_string()}")

    # 2. Find the training vector files
    slum_file, non_slum_file = find_training_files()
    
    # 3. Load and reproject vector files
    slum_gdf = load_and_reproject_vectors(slum_file, target_crs)
    non_slum_gdf = load_and_reproject_vectors(non_slum_file, target_crs)
    
    # 4. Prepare geometries for rasterizing
    # We create a list of (geometry, value) tuples
    
    # Non-Slums will be burned with value 0
    non_slum_shapes = []
    if non_slum_gdf is not None:
        non_slum_shapes = [(geom, 0) for geom in non_slum_gdf.geometry]
        
    # Slums will be burned with value 1
    slum_shapes = []
    if slum_gdf is not None:
        slum_shapes = [(geom, 1) for geom in slum_gdf.geometry]

    if not slum_shapes:
        print("❌ ERROR: No slum geometries were loaded. Cannot create mask.")
        sys.exit(1)
        
    print(f"Found {len(slum_shapes)} slum polygons and {len(non_slum_shapes)} non-slum polygons.")

    # 5. Create the blank raster
    # We fill it with our NO_DATA_VALUE
    mask_data = np.full(out_shape, NO_DATA_VALUE, dtype=rasterio.uint8)

    # 6. Burn the features
    # We burn non-slums first, then slums. This way, if any polygons
    # overlap, the "slum" value (1) will win.
    print("Burning non-slum (Class 0) features...")
    rasterize(non_slum_shapes, out=mask_data, transform=transform, dtype=rasterio.uint8)
    
    print("Burning slum (Class 1) features...")
    rasterize(slum_shapes, out=mask_data, transform=transform, dtype=rasterio.uint8)

    # 7. Update metadata for the new mask file
    out_meta.update({
        "driver": "GTiff",
        "count": 1,              # Only one band
        "dtype": rasterio.uint8, # 8-bit integer
        "nodata": NO_DATA_VALUE  # Set the NoData value
    })
    
    # 8. Write the final mask file
    print(f"Saving ground truth mask to: {OUTPUT_MASK}")
    with rasterio.open(OUTPUT_MASK, 'w', **out_meta) as dest:
        dest.write(mask_data, 1)
        
    print("\n--- Stage 2 Complete ✅ ---")
    print(f"Ground truth file 'mask.tif' is saved in data/training/")

if __name__ == "__main__":
    main()