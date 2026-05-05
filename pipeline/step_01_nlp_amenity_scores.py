"""
Pipeline Step 1: NLP-Based Amenity Quality Scoring
====================================================
Replaces crude (good_reviews - bad_reviews) with VADER sentiment
analysis on actual review text to produce per-category quality scores.

Output: pipeline_output/nlp_amenity_scores.csv
Columns: place_name, Education_Score, Environment_Score, Food_Score,
         Health_Score, Recreation_Score, Religion_Score, Services_Score,
         Transport_Score
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── Initialize VADER ────────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()


def classify_review_file(filename):
    """
    Map a review CSV filename to one of our 8 standard categories.
    Returns None if no match found.
    """
    f_lower = filename.lower()
    for keyword, category in config.REVIEW_FILE_CATEGORY_MAP.items():
        if keyword in f_lower:
            return category
    return None


def compute_category_score_from_reviews(review_files, category):
    """
    Given a list of review CSV file paths for a single category,
    compute a composite NLP quality score (0-10 scale).
    
    Composite = (w1*semantic + w2*rating + w3*volume) * diversity
    """
    all_sentiments = []
    all_ratings = []
    all_amenity_names = set()
    
    for fpath in review_files:
        try:
            df = pd.read_csv(fpath, encoding='utf-8', on_bad_lines='skip')
        except Exception:
            try:
                df = pd.read_csv(fpath, encoding='latin-1', on_bad_lines='skip')
            except Exception:
                continue
        
        if df.empty:
            continue
        
        # ── Collect amenity names for diversity ──
        for col in df.columns:
            if col.lower() in ('name', 'amenity', 'amenity_name', 'place'):
                names = df[col].dropna().unique()
                all_amenity_names.update(names)
                break
        
        # ── Extract ratings ──
        if 'rating' in df.columns:
            ratings = pd.to_numeric(df['rating'], errors='coerce').dropna()
            all_ratings.extend(ratings.tolist())
        
        # ── VADER sentiment on review text ──
        text_cols = [c for c in df.columns if c.lower() in ('text', 'review', 'review_text', 'comment', 'body')]
        if not text_cols:
            # Fallback: use last column that looks like text
            for c in df.columns:
                if df[c].dtype == object and c.lower() not in ('name', 'amenity', 'amenity_name', 'place', 'rating', 'date'):
                    text_cols = [c]
                    break
        
        for col in text_cols:
            texts = df[col].dropna().astype(str)
            for text in texts:
                if len(text.strip()) > 5:  # Skip very short texts
                    vs = analyzer.polarity_scores(text)
                    all_sentiments.append(vs['compound'])
    
    # ── Compute component scores ──
    total_reviews = len(all_sentiments) + len(all_ratings)
    
    if total_reviews == 0:
        return None  # No data → will fallback to distance-based
    
    # 1. Semantic sentiment: VADER compound [-1, 1] → scaled to [0, 10]
    if all_sentiments:
        mean_sentiment = np.mean(all_sentiments)
        semantic_score = (mean_sentiment + 1) * 5  # Maps [-1,1] → [0,10]
    else:
        semantic_score = 5.0  # Neutral default
    
    # 2. Rating score: mean rating [1, 5] → scaled to [0, 10]
    if all_ratings:
        mean_rating = np.mean(all_ratings)
        rating_score = (mean_rating - 1) * 2.5  # Maps [1,5] → [0,10]
    else:
        rating_score = 5.0  # Neutral default
    
    # 3. Volume signal: log(1 + count), capped
    volume_signal = min(np.log1p(total_reviews), config.NLP_VOLUME_CAP)
    
    # 4. Diversity multiplier
    unique_amenities = len(all_amenity_names)
    diversity = min(1 + unique_amenities / 10, config.NLP_DIVERSITY_CAP)
    
    # ── Composite score ──
    raw_score = (
        config.NLP_WEIGHT_SEMANTIC * semantic_score +
        config.NLP_WEIGHT_RATING * rating_score +
        config.NLP_WEIGHT_VOLUME * volume_signal
    ) * diversity
    
    # Clamp to [0, 10]
    final_score = float(np.clip(raw_score, 0, 10))
    
    return final_score


def load_fallback_scores():
    """Load existing fuzzy/distance-based scores as fallback."""
    if os.path.exists(config.FUZZY_SCORES_CSV):
        df = pd.read_csv(config.FUZZY_SCORES_CSV)
        return df
    return pd.DataFrame()


def process_all_places():
    """
    Walk the reviews directory and compute NLP amenity scores
    for every place. Falls back to distance-based scores for
    categories without reviews.
    """
    print("=" * 60)
    print("  STEP 1: NLP-Based Amenity Quality Scoring")
    print("=" * 60)
    
    fallback_df = load_fallback_scores()
    
    # Get all place names from data directory (superset)
    all_places = set()
    if os.path.exists(config.DATA_DIR):
        all_places.update(d for d in os.listdir(config.DATA_DIR) 
                         if os.path.isdir(os.path.join(config.DATA_DIR, d)))
    if os.path.exists(config.REVIEWS_DIR):
        all_places.update(d for d in os.listdir(config.REVIEWS_DIR)
                         if os.path.isdir(os.path.join(config.REVIEWS_DIR, d)))
    
    # Also include places from fuzzy scores
    if not fallback_df.empty and 'place_name' in fallback_df.columns:
        all_places.update(fallback_df['place_name'].tolist())
    
    all_places = sorted(all_places)
    print(f"\nFound {len(all_places)} total places to process.\n")
    
    results = []
    
    for place in all_places:
        place_scores = {'place_name': place}
        review_dir = os.path.join(config.REVIEWS_DIR, place)
        
        # ── Group review files by category ──
        category_files = {cat: [] for cat in config.AMENITY_CATEGORIES}
        
        if os.path.exists(review_dir):
            for f in os.listdir(review_dir):
                if f.endswith('.csv'):
                    cat = classify_review_file(f)
                    if cat and cat in category_files:
                        category_files[cat].append(os.path.join(review_dir, f))
        
        # ── Compute NLP score per category ──
        has_any_reviews = False
        for cat in config.AMENITY_CATEGORIES:
            col_name = f"{cat}_Score"
            
            if category_files[cat]:
                nlp_score = compute_category_score_from_reviews(category_files[cat], cat)
                if nlp_score is not None:
                    place_scores[col_name] = round(nlp_score, 2)
                    has_any_reviews = True
                    continue
            
            # Fallback to distance-based score
            if not fallback_df.empty and place in fallback_df['place_name'].values:
                row = fallback_df[fallback_df['place_name'] == place].iloc[0]
                if col_name in row.index:
                    place_scores[col_name] = float(row[col_name])
                else:
                    place_scores[col_name] = 0.0
            else:
                place_scores[col_name] = 0.0
        
        status = "NLP" if has_any_reviews else "fallback"
        print(f"  [{status:>8}] {place}")
        results.append(place_scores)
    
    # ── Save output ──
    df_out = pd.DataFrame(results)
    
    # Ensure all score columns exist
    for cat in config.AMENITY_CATEGORIES:
        col = f"{cat}_Score"
        if col not in df_out.columns:
            df_out[col] = 0.0
    
    # Reorder columns
    cols = ['place_name'] + [f"{cat}_Score" for cat in config.AMENITY_CATEGORIES]
    df_out = df_out[cols]
    
    df_out.to_csv(config.NLP_AMENITY_SCORES_CSV, index=False)
    
    print(f"\n✅ NLP amenity scores saved to: {config.NLP_AMENITY_SCORES_CSV}")
    print(f"   Total places: {len(df_out)}")
    print(f"   Score ranges:")
    for col in [c for c in df_out.columns if c != 'place_name']:
        vals = df_out[col]
        print(f"     {col:>20}: min={vals.min():.2f}, mean={vals.mean():.2f}, max={vals.max():.2f}")
    
    return df_out


if __name__ == "__main__":
    process_all_places()
