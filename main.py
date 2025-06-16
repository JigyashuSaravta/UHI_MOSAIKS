import os
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from planetary_computer import sign
import pystac_client
import stackstac
import rioxarray
import xarray as xr
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def extract_satellite_features(locations,
                               datetime="2021-07-24",
                               satellite="sentinel-2-l2a",
                               bands=None,
                               resolution=10,
                               cloud_cover_threshold=20):
    """
    Extract satellite reflectance values from Microsoft Planetary Computer.
    """
    if bands is None:
        bands = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR

    print(f"[Debug] Locations sample: {locations[:3]}")

    # Step 1: Convert to GeoDataFrame
    print("[Step 1] Converting locations to GeoDataFrame")
    gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in locations], crs="EPSG:4326")

    # Step 2: Connect to STAC API
    print("[Step 2] Connecting to Planetary Computer STAC API")
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # Step 3: STAC Search with fixed bbox
    print("[Step 3] Running STAC search with fixed NYC bounding box")
    bbox = [-74.03, 40.69, -73.9, 40.88]
    print(f"[Debug] Bounding box used: {bbox}")
    search = stac.search(
        bbox=bbox,
        datetime=datetime,
        collections=[satellite],
        query={"eo:cloud_cover": {"lt": cloud_cover_threshold}}
    )

    items = list(search.get_items())
    print(f"[Step 3a] Found {len(items)} matching items")
    if not items:
        raise ValueError("No matching satellite imagery found.")

    # Step 4: Sign items
    print("[Step 4] Signing STAC items")
    signed_items = [sign(item) for item in items]
    print(f"[Debug] First item asset keys: {list(signed_items[0].assets.keys())}")

    # Step 5: Stack bands
    print("[Step 5] Stacking assets using stackstac")
    try:
        data = stackstac.stack(
            signed_items,
            assets=bands,
            resolution=resolution,
            epsg=32618  # UTM zone for NYC
        )
        print(f"[Debug] Stacked dimensions: {data.dims}")
        print(f"[Debug] Stacked shape: {data.shape}")
        print(f"[Debug] time size: {data.sizes.get('time', 'N/A')}")
        print(f"[Debug] band size: {data.sizes.get('band', 'N/A')}")

        if data.sizes.get("time", 0) == 0 or data.sizes.get("band", 0) == 0:
            raise ValueError("Stacked data has empty dimensions.")
    except Exception as e:
        raise ValueError(f"stackstac.stack failed: {e}")

    # Step 6: Project coordinates and sample reflectance
    gdf_proj = gdf.to_crs(data.rio.crs)
    coords = [(geom.x, geom.y) for geom in gdf_proj.geometry]

    # Select first time slice (band, y, x)
    data_single_time = data.isel(time=0).transpose("band", "y", "x")

    # Sample at locations
    sampled = data_single_time.sel(
        x=[x for x, y in coords],
        y=[y for x, y in coords],
        method="nearest"
    )

    # Convert to DataFrame using xarray machinery
    df = sampled.to_dataframe(name="value").reset_index()

    # Pivot: we want rows = points, columns = bands
    df = df.pivot(index=["x", "y"], columns="band", values="value").reset_index(drop=True)
    df.columns = [f"band_{b}" for b in bands]

    return df


def batch_extract_features_with_cache(df,
                                      chunk_size=500,
                                      cache_dir="features_cache",
                                      max_retries=3,
                                      retry_delay=10,
                                      **kwargs):
    """
    Batch feature extractor with CSV caching and retry logic.
    """
    os.makedirs(cache_dir, exist_ok=True)
    all_features = []

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunk_idx = i // chunk_size
        cache_path = os.path.join(cache_dir, f"chunk_{chunk_idx}.csv")

        if os.path.exists(cache_path):
            print(f"[Cache] Loaded chunk {chunk_idx}")
            chunk_features = pd.read_csv(cache_path)
        else:
            print(f"[Fetch] Processing chunk {chunk_idx}")
            locations = list(zip(chunk["latitude"], chunk["longitude"]))

            for attempt in range(1, max_retries + 1):
                try:
                    chunk_features = extract_satellite_features(locations, **kwargs)
                    chunk_features.to_csv(cache_path, index=False)
                    print(f"[Success] Chunk {chunk_idx} saved")
                    break
                except Exception as e:
                    print(f"[Error] Attempt {attempt} for chunk {chunk_idx} failed: {e}")
                    if attempt == max_retries:
                        print(f"[Fail] Skipping chunk {chunk_idx}")
                        chunk_features = pd.DataFrame()
                    else:
                        time.sleep(retry_delay)

        all_features.append(chunk_features)

    return pd.concat(all_features, ignore_index=True)
