import sys
import os
import glob
import rasterio
from rasterio.mask import mask
import geopandas as gpd

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
AOI_FILE = os.path.join(PROJECT_ROOT, 'data', 'aoi', 'aoi.geojson')

# Create the processed directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define the 10m bands we need for our analysis
# B02=Blue, B03=Green, B04=Red, B08=NIR
BANDS_10M = ['B02', 'B03', 'B04', 'B08']

# --- Main Script ---

def find_safe_folder():
    """Find the first .SAFE folder in the raw data directory."""
    print(f"Searching for .SAFE folder in: {RAW_DIR}")
    safe_folders = glob.glob(os.path.join(RAW_DIR, 'S2*.SAFE'))
    
    if not safe_folders:
        print("❌ ERROR: No .SAFE folder found in data/raw/")
        print("Please make sure you have unzipped your download and placed the .SAFE folder in data/raw/")
        sys.exit(1)
    
    if len(safe_folders) > 1:
        print(f"⚠️ Warning: Found {len(safe_folders)} .SAFE folders. Using the first one:")
        print(f"   {safe_folders[0]}")
    
    print(f"✅ Found product folder: {os.path.basename(safe_folders[0])}")
    return safe_folders[0]

def find_band_files(safe_folder):
    """Find the full paths to the 10m band .jp2 files."""
    band_paths = {}
    print("Locating 10m band files (B02, B03, B04, B08)...")
    
    for band in BANDS_10M:
        # Use recursive glob to find the band file deep in the complex .SAFE structure
        # The path is usually .../GRANULE/.../IMG_DATA/R10m/T..._B0X_10m.jp2
        search_pattern = os.path.join(safe_folder, '**', f'*_{band}_10m.jp2')
        try:
            band_file = glob.glob(search_pattern, recursive=True)[0]
            band_paths[band] = band_file
        except IndexError:
            print(f"❌ ERROR: Could not find band {band} in {safe_folder}")
            sys.exit(1)
            
    print("✅ All 10m band files located.")
    return band_paths

def clip_and_stack_bands(band_paths, aoi_file, output_filename):
    """Clips all bands to the AOI, stacks them, and saves as a GeoTIFF."""
    
    # 1. Read the AOI GeoJSON
    print(f"Reading AOI from: {aoi_file}")
    aoi_gdf = gpd.read_file(aoi_file)
    
    clipped_bands_data = []
    out_meta = None
    out_transform = None

    print("Clipping and stacking bands:")
    for band_name in BANDS_10M: # Iterate in order
        band_path = band_paths[band_name]
        print(f"  Processing {band_name}...")
        
        with rasterio.open(band_path) as src:
            # 2. Reproject AOI to match the raster's CRS
            # This is critical for clipping
            if aoi_gdf.crs != src.crs:
                print(f"     Reprojecting AOI from {aoi_gdf.crs} to {src.crs}...")
                aoi_reprojected = aoi_gdf.to_crs(src.crs)
            else:
                aoi_reprojected = aoi_gdf
            
            # 3. Clip the raster using the AOI geometry
            try:
                out_image, out_transform = mask(src, aoi_reprojected.geometry, crop=True)
            except ValueError as e:
                print(f"❌ ERROR: Clipping failed. {e}")
                print("This can happen if your AOI is outside the image bounds.")
                sys.exit(1)

            # 4. Store the metadata from the first band
            if out_meta is None:
                out_meta = src.meta.copy()

            # Add the clipped band (1st dim) to our list
            clipped_bands_data.append(out_image[0])

    print("✅ Clipping complete.")

    # 5. Update metadata for the new stacked file
    out_meta.update({
        "driver": "GTiff",
        "height": clipped_bands_data[0].shape[0],
        "width": clipped_bands_data[0].shape[1],
        "transform": out_transform,
        "count": len(clipped_bands_data), # Number of bands
        "dtype": clipped_bands_data[0].dtype
    })

    # 6. Write the stacked GeoTIFF
    print(f"Writing stacked GeoTIFF to: {output_filename}")
    with rasterio.open(output_filename, "w", **out_meta) as dest:
        for i, band_data in enumerate(clipped_bands_data, start=1):
            dest.write(band_data, i)
            
    print("✅ Stacked file saved successfully.")


def main():
    print("--- Stage 2: Data Preprocessing ---")
    
    # 1. Find the downloaded data
    safe_folder = find_safe_folder()
    
    # 2. Find the paths to the 10m band files
    band_paths = find_band_files(safe_folder)
    
    # 3. Define an output filename
    safe_name = os.path.basename(safe_folder).replace('.SAFE', '')
    output_filename = os.path.join(PROCESSED_DIR, f"{safe_name}_4BANDS_clipped.tif")
    
    # 4. Run the processing
    clip_and_stack_bands(band_paths, AOI_FILE, output_filename)
    
    print("\n--- Stage 2 Complete ✅ ---")
    print(f"Your processed file is ready at: {output_filename}")

if __name__ == "__main__":
    main()