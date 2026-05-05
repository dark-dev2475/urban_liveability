"""
Pipeline Step 3: Build Master Dataset
=======================================
Merges NLP amenity scores + NLP context scores + labels
into a single clean training dataset.

Output: pipeline_output/master_dataset.csv
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def normalize_place_name(name):
    """Normalize place names for matching across datasets."""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    # Remove common suffixes for matching
    name = name.replace('_anemities', '').replace('_amenities', '')
    return name


def build_master_dataset():
    """
    Merge all feature sources + labels into master dataset.
    """
    print("=" * 60)
    print("  STEP 3: Build Master Dataset")
    print("=" * 60)
    
    # ── Load NLP amenity scores ──
    if not os.path.exists(config.NLP_AMENITY_SCORES_CSV):
        print("❌ NLP amenity scores not found. Run step_01 first.")
        return None
    amenity_df = pd.read_csv(config.NLP_AMENITY_SCORES_CSV)
    print(f"  Loaded amenity scores: {len(amenity_df)} places")
    
    # ── Load NLP context scores ──
    if not os.path.exists(config.NLP_CONTEXT_SCORES_CSV):
        print("❌ NLP context scores not found. Run step_02 first.")
        return None
    context_df = pd.read_csv(config.NLP_CONTEXT_SCORES_CSV)
    print(f"  Loaded context scores: {len(context_df)} places")
    
    # ── Load labels ──
    if not os.path.exists(config.LABELS_CSV):
        print("❌ Labels file not found.")
        return None
    labels_df = pd.read_csv(config.LABELS_CSV)
    # Clean: drop empty rows
    labels_df = labels_df.dropna(subset=['place_name', 'is_slum'])
    labels_df['is_slum'] = labels_df['is_slum'].astype(int)
    print(f"  Loaded labels: {len(labels_df)} places")
    
    # ── Handle Environment_Score ──
    # The amenity scores have Environment_Score from reviews (mostly 0 or fallback)
    # The context scores have Environment_Score from news
    # Strategy: use news-based environment score (it's more informative)
    
    # If context_df has Environment_Score, use it; otherwise keep amenity's
    if 'Environment_Score' in context_df.columns:
        # Remove Environment_Score from amenity_df to avoid collision
        if 'Environment_Score' in amenity_df.columns:
            amenity_df = amenity_df.drop(columns=['Environment_Score'])
    
    # ── Merge: start with labels, left-join features ──
    master = labels_df.copy()
    master = pd.merge(master, amenity_df, on='place_name', how='left')
    master = pd.merge(master, context_df, on='place_name', how='left')
    
    # Merge satellite scores
    if hasattr(config, 'SATELLITE_SCORES_CSV') and os.path.exists(config.SATELLITE_SCORES_CSV):
        sat_df = pd.read_csv(config.SATELLITE_SCORES_CSV)
        print(f"  Loaded satellite scores: {len(sat_df)} places")
        master = pd.merge(master, sat_df, on='place_name', how='left')
    else:
        print("  ⚠ Satellite scores not found")
        master['Satellite_Score'] = 0.0
    
    # ── Fill missing values ──
    # For amenity scores: 0 means no data
    for col in config.AMENITY_SCORE_COLUMNS:
        if col in master.columns:
            master[col] = master[col].fillna(0.0)
    
    # For context scores: 0 means neutral
    for col in config.CONTEXT_SCORE_COLUMNS:
        if col in master.columns:
            master[col] = master[col].fillna(0.0)
    
    if 'Environment_Score' in master.columns:
        master['Environment_Score'] = master['Environment_Score'].fillna(3.0)
    
    # Remove duplicate columns (e.g. Environment_Score.1 if it happened)
    master = master.loc[:, ~master.columns.duplicated()]

    # ── Select final columns ──
    final_cols = ['place_name', 'is_slum'] + config.ALL_FEATURE_COLUMNS
    # Add Environment_Score if present and not already in ALL_FEATURE_COLUMNS
    available_cols = [c for c in final_cols if c in master.columns]
    master = master[available_cols]
    
    # Add is_synthetic flag
    master['is_synthetic'] = False
    
    # ── Drop any remaining NaN rows ──
    before = len(master)
    master = master.dropna()
    after = len(master)
    if before != after:
        print(f"  ⚠ Dropped {before - after} rows with NaN values")
    
    # ── Save ──
    master.to_csv(config.MASTER_DATASET_CSV, index=False)
    
    print(f"\n✅ Master dataset saved to: {config.MASTER_DATASET_CSV}")
    print(f"   Total samples: {len(master)}")
    print(f"   Class distribution:")
    print(f"     Liveable (0): {(master['is_slum'] == 0).sum()}")
    print(f"     Slum     (1): {(master['is_slum'] == 1).sum()}")
    print(f"\n   Feature statistics:")
    for col in [c for c in master.columns if c not in ('place_name', 'is_slum', 'is_synthetic')]:
        vals = pd.to_numeric(master[col], errors='coerce').dropna()
        if len(vals) > 0:
            print(f"     {col:>25}: min={vals.min():.2f}, mean={vals.mean():.2f}, max={vals.max():.2f}, std={vals.std():.2f}")
    
    return master


if __name__ == "__main__":
    build_master_dataset()
