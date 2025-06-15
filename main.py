import os
import time
import pandas as pd
from mosaiks import get_features

def extract_mosaiks_features(locations,
                              datetime="2021-07-24",
                              satellite_name="sentinel-2-l2a",
                              image_resolution=10,
                              image_bands=["B02", "B03", "B04", "B08"],  # RGB + NIR for Sentinel-2
                              image_width=3000,
                              n_mosaiks_features=4000,
                              parallelize=False):
    """
    Extract MOSAIKS features using updated API signature.
    """
    lats, lons = zip(*locations)

    features_df = get_features(
        latitudes=list(lats),
        longitudes=list(lons),
        datetime=datetime,
        satellite_name=satellite_name,
        image_resolution=image_resolution,
        image_bands=image_bands,
        image_width=image_width,
        n_mosaiks_features=n_mosaiks_features,
        parallelize=parallelize,
        model_device="cpu"
    )

    return features_df


def batch_extract_features_with_cache(df, chunk_size=500, cache_dir="features_cache",
                                      max_retries=3, retry_delay=10):
    """
    Extract MOSAIKS features for large dataset using batching, caching, and retry logic.

    Args:
        df (pd.DataFrame): Input with 'latitude' and 'longitude'.
        chunk_size (int): Number of locations per batch.
        cache_dir (str): Directory to store cached results.
        max_retries (int): How many times to retry on failure.
        retry_delay (int): Seconds to wait before retrying.

    Returns:
        pd.DataFrame: All features concatenated.
    """
    os.makedirs(cache_dir, exist_ok=True)
    all_features = []

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunk_idx = i // chunk_size
        cache_path = os.path.join(cache_dir, f"chunk_{chunk_idx}.csv")

        if os.path.exists(cache_path):
            print(f"[Cache] Loading chunk {chunk_idx}")
            chunk_features = pd.read_csv(cache_path)
        else:
            print(f"[Fetch] Processing chunk {chunk_idx}...")
            locations = list(zip(chunk["latitude"], chunk["longitude"]))

            for attempt in range(1, max_retries + 1):
                try:
                    features = extract_mosaiks_features(locations)
                    chunk_features = pd.DataFrame(features)
                    chunk_features.to_csv(cache_path, index=False)
                    print(f"[Success] Chunk {chunk_idx} saved to cache.")
                    break
                except Exception as e:
                    print(f"[Error] Attempt {attempt} failed: {e}")
                    if attempt == max_retries:
                        print(f"[Fail] Skipping chunk {chunk_idx} after {max_retries} failed attempts.")
                        chunk_features = pd.DataFrame()  # empty fallback
                    else:
                        time.sleep(retry_delay)

        all_features.append(chunk_features)

    return pd.concat(all_features, ignore_index=True)
