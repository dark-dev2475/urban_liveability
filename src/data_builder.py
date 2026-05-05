"""
dataset_builder.py

Scans data/<area>/ folders, extracts ML features per area using the same
accessibility logic you already have, uses the fuzzy model to generate a
target liveability score, and writes output/training_dataset.csv.

Depends on: fuzzy_liveability.py (must be in same folder or importable).
"""

import os
import pandas as pd
import numpy as np

# Import directly from your existing fuzzy model (must be accessible)
from fuzzy_liveability import (
    load_location_data,
    calculate_category_score,
    calculate_category_score_geocoded,
    calculate_air_quality_score,
    evaluate_liveability,
    define_fuzzy_system,
    haversine_distance  # if defined in fuzzy_liveability
)

# Output path (relative to project root)
OUTPUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "training_dataset.csv"))
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)


# ---------- Feature helpers ----------

def safe_count(df):
    return 0 if df is None or df.empty else len(df)

def safe_unique_amenities(df):
    if df is None or df.empty:
        return 0
    return int(df['Amenity'].nunique()) if 'Amenity' in df.columns else 0

def compute_distance_series_from_centroid(df):
    """Return a Series of distances (km) from centroid if lat/lon exist; else None"""
    if df is None or df.empty:
        return None
    if 'Distance_km' in df.columns and df['Distance_km'].notna().any():
        return df['Distance_km'].dropna()
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        center_lat = df['Latitude'].mean()
        center_lon = df['Longitude'].mean()
        distances = df.apply(
            lambda r: haversine_distance(center_lat, center_lon, r['Latitude'], r['Longitude'])
            if pd.notna(r.get('Latitude')) and pd.notna(r.get('Longitude')) else np.nan,
            axis=1
        )
        return distances.dropna()
    return None

def compute_distance_features(df):
    """
    Returns dict:
      - count
      - unique_amenities
      - min_distance
      - mean_distance
      - max_distance
      - decay_mean (mean of exp(-0.25 * d))
      - top5_decay_sum
    """
    out = {
        'count': 0,
        'unique_amenities': 0,
        'min_distance': np.nan,
        'mean_distance': np.nan,
        'max_distance': np.nan,
        'decay_mean': 0.0,
        'top5_decay_sum': 0.0
    }

    if df is None or df.empty:
        return out

    out['count'] = safe_count(df)
    out['unique_amenities'] = safe_unique_amenities(df)

    dist_series = compute_distance_series_from_centroid(df)
    if dist_series is None or dist_series.empty:
        return out

    out['min_distance'] = float(dist_series.min())
    out['mean_distance'] = float(dist_series.mean())
    out['max_distance'] = float(dist_series.max())

    # decay values
    decay = np.exp(-0.25 * dist_series.clip(lower=0))
    out['decay_mean'] = float(decay.mean())
    out['top5_decay_sum'] = float(decay.nlargest(5).sum())

    return out


# ---------- Single-area feature builder ----------

CATEGORIES = ["Transport", "Health", "Education", "Services", "Recreation", "Food", "Environment", "Religion"]

def build_features_for_area(area_name, area_path, fuzzy_system=None):
    """
    Load category CSVs for area_name from area_path (folder) and build a single-row feature dict.
    """
    # Use your loader (it matches file names using keywords and location prefix)
    location_data = load_location_data(area_name, area_path)

    # category scores (existing accessibility function) - keep for backward compatibility
    category_scores = {}
    # more ML-friendly features
    features = {'area': area_name}

    for cat in CATEGORIES:
        df = location_data.get(cat, pd.DataFrame())
        # legacy category-level "accessibility score" using your function (0-10 style)
        try:
            legacy_score = calculate_category_score(df, cat)
        except Exception:
            legacy_score = 0.0
        category_scores[f"{cat}_legacy_score"] = float(legacy_score)

        # more features
        df_feats = compute_distance_features(df)
        # prefix the features with category
        features[f"{cat}_count"] = df_feats['count']
        features[f"{cat}_unique_amenities"] = df_feats['unique_amenities']
        features[f"{cat}_min_distance"] = df_feats['min_distance']
        features[f"{cat}_mean_distance"] = df_feats['mean_distance']
        features[f"{cat}_max_distance"] = df_feats['max_distance']
        features[f"{cat}_decay_mean"] = df_feats['decay_mean']
        features[f"{cat}_top5_decay_sum"] = df_feats['top5_decay_sum']

        # Environment: also include air quality score if available (uses your function)
        if cat == "Environment":
            try:
                aq = calculate_air_quality_score(df)
            except Exception:
                aq = 0.0
            features['Environment_air_quality'] = float(aq)

    # Run fuzzy model to generate target label (0-100)
    if fuzzy_system is None:
        fuzzy_system = define_fuzzy_system()

    try:
        fuzzy_result = evaluate_liveability(location_data, fuzzy_system)
        liveability_target = float(fuzzy_result.get('overall_score', 0.0))
    except Exception:
        liveability_target = 0.0

    # Attach legacy scores (optional) and the target label
    features.update(category_scores)
    features['liveability'] = liveability_target

    return features


# ---------- Build dataset across all areas ----------

def build_ml_dataset(base_data_dir=None):
    """
    Scans the data directory for area folders and builds training CSV.
    """
    if base_data_dir is None:
        # project_root/data
        base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    if not os.path.isdir(base_data_dir):
        raise FileNotFoundError(f"Data directory not found: {base_data_dir}")

    areas = [
        d for d in os.listdir(base_data_dir)
        if os.path.isdir(os.path.join(base_data_dir, d))
    ]
    if not areas:
        raise ValueError(f"No area folders found in data directory: {base_data_dir}")

    rows = []
    fuzzy_system = define_fuzzy_system()

    for area in areas:
        area_path = os.path.join(base_data_dir, area)
        print(f"Building features for: {area} (path={area_path})")
        feats = build_features_for_area(area, area_path, fuzzy_system=fuzzy_system)
        rows.append(feats)

    df = pd.DataFrame(rows)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved training dataset to: {OUTPUT_CSV} (rows={len(df)})")
    return df


# ---------- Main ----------

if __name__ == "__main__":
    # default base data directory (project_root/data)
    base_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    build_ml_dataset(base_data)
