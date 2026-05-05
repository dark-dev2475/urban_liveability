import sys
import os
import json
import requests
import geopandas as gpd
from shapely.geometry import shape
from datetime import datetime

# --- Configuration ---
COPERNICUS_USER = 'gagan.rkvvm@gmail.com'
COPERNICUS_PASS = 'Gagan@2475678'

TOKEN_URL = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
CATALOG_URL = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AOI_FILE = os.path.join(PROJECT_ROOT, 'data', 'aoi', 'aoi.geojson')
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# --- Helper Functions ---

def get_token():
    """Authenticate and get access token from Copernicus Data Space."""
    print("Authenticating with Copernicus Data Space...")
    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": COPERNICUS_USER,
        "password": COPERNICUS_PASS
    }
    try:
        response = requests.post(TOKEN_URL, data=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        print("✅ Authentication successful.")
        return response.json()["access_token"]
    except requests.exceptions.HTTPError as e:
        print(f"❌ Authentication failed: {e.response.status_code} - {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred during authentication: {e}")
        sys.exit(1)


def read_aoi():
    """Read AOI GeoJSON and return WKT footprint."""
    try:
        aoi = gpd.read_file(AOI_FILE)
        # Ensure it's in WGS84 (EPSG:4326) as required by the API
        if aoi.crs.to_epsg() != 4326:
            aoi = aoi.to_crs(epsg=4326)
        footprint = aoi.geometry.union_all().wkt
        print(f"✅ Successfully read AOI file: {AOI_FILE}")
        return footprint
    except Exception as e:
        print(f"❌ Error reading AOI file: {e}")
        sys.exit(1)


def search_products(token, footprint, year="2024"):
    """Query the CDSE catalog for Sentinel-2 L2A products for the AOI."""
    print(f"🔍 Querying for Sentinel-2 L2A products in {year} for your AOI...")
    headers = {"Authorization": f"Bearer {token}"}

    # --- THIS IS THE CORRECTED QUERY ---
    params = {
        "$filter": (
            f"(Collection/Name eq 'SENTINEL-2') and "
            # FIX 1: Added the spatial intersection with your AOI footprint
            f"(OData.CSC.Intersects(area=geography'{footprint}')) and "
            # FIX 2: ONLY search for L2A products. Removed L1C.
            f"(startswith(Name,'S2A_MSIL2A_') or startswith(Name,'S2B_MSIL2A_')) and "
            f"ContentDate/Start ge {year}-01-01T00:00:00Z and "
            f"ContentDate/End le {year}-12-31T23:59:59Z"
        ),
        "$top": 20, # Get the 20 most recent products matching the filter
        "$orderby": "ContentDate/Start desc"
    }
    # --- END OF CORRECTION ---

    try:
        response = requests.get(CATALOG_URL, headers=headers, params=params)
        response.raise_for_status()
        products = response.json().get("value", [])
        
        if not products:
            print(f"⚠️ No products found for {year}.")
        else:
            print(f"✅ Found {len(products)} Sentinel-2 L2A products for {year}.")
        return products
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ Query failed ({year}): {e.response.status_code} - {e.response.text}")
        return []
    except Exception as e:
        print(f"❌ An error occurred during search ({year}): {e}")
        return []


def filter_by_cloud_cover(products, max_cloud=30):
    """Filter products locally by cloud cover percentage."""
    low_cloud_products = []
    for p in products:
        cloud_val = None
        for attr in p.get("Attributes", []):
            if attr.get("Name") == "cloudCoverPercentage":
                val_raw = (
                    attr.get("Value")
                    or attr.get("OData.CSC.DoubleAttribute", {}).get("Value")
                )
                if val_raw is not None:
                    try:
                        cloud_val = float(val_raw)
                    except ValueError:
                        cloud_val = None
                break # Found the cloud cover attribute
        
        if cloud_val is not None and cloud_val <= max_cloud:
            low_cloud_products.append((p, cloud_val))
            
    if not low_cloud_products:
        print(f"⚠️ No products found with cloud cover ≤ {max_cloud}%.")
    else:
        print(f"✅ Found {len(low_cloud_products)} products with cloud cover ≤ {max_cloud}%.")
    return low_cloud_products


def download_product(token, product_id, product_name):
    """Try downloading the product using CDSE Zipper; skip if not archived."""
    headers = {"Authorization": f"Bearer {token}"}
    # Use the "Products('...')" OData syntax to get the URL
    product_meta_url = f"{CATALOG_URL}('{product_id}')"
    
    try:
        # First, check if the product is "Online"
        meta_resp = requests.get(product_meta_url, headers=headers)
        meta_resp.raise_for_status()
        is_online = meta_resp.json().get('Online', False)
        
        if not is_online:
            print(f"⚠️ Product {product_name} is 'Offline' (Long Term Archive). Skipping...")
            return None

        # Product is Online, proceed to download via zipper
        download_request_url = f"https://zipper.dataspace.copernicus.eu/api/v1/download/{product_id}"
        resp = requests.get(download_request_url, headers=headers)
        resp.raise_for_status()

        download_url = resp.json().get("url")
        if not download_url:
            print(f"❌ No download URL received for {product_name}.")
            return None

        print(f"🔗 Download URL received, starting download for {product_name} ...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        output_path = os.path.join(DOWNLOAD_DIR, f"{product_name}.zip")
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = (downloaded / total_size * 100) if total_size > 0 else 0
                    print(f"\r📦 Downloading... {percent:.1f}%", end="")

        print(f"\n✅ Download complete: {output_path}")
        return output_path

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
             print(f"⚠️ Product {product_name} not found in download API. Skipping...")
        else:
            print(f"❌ Failed to download {product_name}: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        print(f"❌ An error occurred downloading {product_name}: {e}")
        return None


def save_metadata(metadata, filepath):
    """Save product metadata to a JSON file."""
    meta_path = filepath.replace(".zip", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"📝 Metadata saved at: {meta_path}")


# --- Main ---
def main():
    print("--- Stage 1: Data Sourcing ---")

    # 1. Authenticate
    token = get_token()

    # 2. Read AOI
    footprint = read_aoi()

    # 3. Multi-year search sequence
    #    Start with most recent year first, as it's most likely to be "Online"
    years_to_try = ["2024", "2023", "2022", "2021", "2020"]
    all_products = []
    active_year = None

    for year in years_to_try:
        products = search_products(token, footprint, year=year)
        if products:
            all_products = products
            active_year = year
            print(f"🛰 Using products from {active_year}")
            break # Stop searching as soon as we find products

    if not all_products:
        print("❌ No Sentinel-2 L2A products found for your AOI in any target year.")
        sys.exit(0)

    # 4. Filter by cloud cover
    products_to_try = filter_by_cloud_cover(all_products, max_cloud=10)
    if not products_to_try:
        print("⚠️ No low-cloud (<=10%) images found, trying ≤ 30%...")
        products_to_try = filter_by_cloud_cover(all_products, max_cloud=30)
        if not products_to_try:
            print("🌧 No clear images (<=30%) found. Proceeding with all scenes from this year...")
            # --- THIS IS THE CORRECTED FALLBACK ---
            # Try all products from the query, not just the first one
            products_to_try = [(p, None) for p in all_products] # fallback
            # --- END OF CORRECTION ---

    # 5. Try downloading sequentially until one works
    output_path = None
    downloaded_product = None
    
    print(f"\nAttempting to download {len(products_to_try)} potential product(s)...")
    
    for p, cloud_val in products_to_try:
        pid = p["Id"]
        pname = p["Name"]
        print(f"⬇️  Trying product: {pname} (Cloud: {cloud_val if cloud_val is not None else 'N/A'})")
        output_path = download_product(token, pid, pname)
        if output_path:
            downloaded_product = (p, cloud_val)
            break # Success! Stop trying.

    if not output_path:
        print("❌ No downloadable 'Online' product found. All suitable products may be in Long Term Archive.")
        sys.exit(0)

    # 6. Save metadata
    best_product, cloud_val = downloaded_product
    product_id = best_product["Id"]
    product_name = best_product["Name"]
    product_date = best_product["ContentDate"]["Start"]

    metadata = {
        "id": product_id,
        "name": product_name,
        "date": product_date,
        "cloud_cover": cloud_val,
        "download_path": output_path,
        "downloaded_at": datetime.utcnow().isoformat() + "Z"
    }
    save_metadata(metadata, output_path)

    print("\n--- Stage 1 Complete ✅ ---")


if __name__ == "__main__":
    main()