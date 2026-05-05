"""
Pipeline Step 2: NLP-Based Contextual / Social Scoring
========================================================
Analyzes news articles with VADER + keyword severity weighting
to extract Crime, Sanitation, Social Stress, and Environment scores.
Normalizes by article count to remove coverage-volume bias.

Also computes an overall Sentiment_Score from review + news data.

Output: pipeline_output/nlp_context_scores.csv
Columns: place_name, Sentiment_Score, Crime_Score, Sanitation_Score,
         Social_Stress_Score, Environment_Score
"""

import os
import sys
import re
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def compute_keyword_score(text, keyword_dict):
    """
    For a single text, compute weighted keyword score.
    Returns (raw_keyword_score, sentiment_compound).
    """
    text_lower = text.lower()
    score = 0.0
    for keyword, weight in keyword_dict.items():
        # Use word-boundary matching
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text_lower, re.IGNORECASE):
            score += weight
    
    vs = analyzer.polarity_scores(text)
    compound = vs['compound']
    
    return score, compound


def compute_environment_score_from_articles(articles_df):
    """
    Extract environment score from news articles.
    Positive keywords indicate good environment, negative indicate bad.
    Score: 0 (very polluted) to 10 (very clean/green).
    """
    if articles_df.empty:
        return None
    
    total_neg = 0.0
    total_pos = 0.0
    n_articles = len(articles_df)
    
    for _, row in articles_df.iterrows():
        text = f"{row.get('title', '')} {row.get('snippet', '')}"
        text_lower = text.lower()
        
        # Negative environment signals
        for kw, weight in config.ENVIRONMENT_KEYWORDS_NEGATIVE.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                total_neg += weight
        
        # Positive environment signals
        for kw, weight in config.ENVIRONMENT_KEYWORDS_POSITIVE.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                total_pos += weight
    
    # Normalize by article count
    neg_per_article = total_neg / max(n_articles, 1)
    pos_per_article = total_pos / max(n_articles, 1)
    
    # Environment score: start at 5 (neutral), go up with positive, down with negative
    env_score = 5.0 + (pos_per_article - neg_per_article) * 0.5
    return float(np.clip(env_score, 0, 10))


def process_news_for_place(news_filepath):
    """
    Process a single place's news CSV and return all scores.
    """
    try:
        df = pd.read_csv(news_filepath, encoding='utf-8', on_bad_lines='skip')
    except Exception:
        try:
            df = pd.read_csv(news_filepath, encoding='latin-1', on_bad_lines='skip')
        except Exception:
            return None
    
    if df.empty:
        return None
    
    n_articles = len(df)
    
    # ── Build full text column ──
    texts = []
    for _, row in df.iterrows():
        title = str(row.get('title', ''))
        snippet = str(row.get('snippet', ''))
        full = f"{title} {snippet}".strip()
        if len(full) > 5:
            texts.append(full)
    
    if not texts:
        return None
    
    n_articles = len(texts)
    
    # ── Compute article-level scores ──
    crime_total = 0.0
    sanitation_total = 0.0
    stress_total = 0.0
    sentiment_compounds = []
    
    for text in texts:
        vs = analyzer.polarity_scores(text)
        compound = vs['compound']
        sentiment_compounds.append(compound)
        
        # Crime: keyword score amplified by negative sentiment
        crime_kw_score, _ = compute_keyword_score(text, config.CRIME_KEYWORDS)
        # Amplify crime signal when sentiment is negative
        amplifier = 1 + max(0, -compound)  # 1.0 to 2.0
        crime_total += crime_kw_score * amplifier
        
        # Sanitation: same approach
        san_kw_score, _ = compute_keyword_score(text, config.SANITATION_KEYWORDS)
        sanitation_total += san_kw_score * amplifier
        
        # Social Stress
        stress_kw_score, _ = compute_keyword_score(text, config.SOCIAL_STRESS_KEYWORDS)
        stress_total += stress_kw_score * amplifier
    
    # ── Normalize by article count (KEY improvement) ──
    crime_score = crime_total / n_articles
    sanitation_score = sanitation_total / n_articles
    stress_score = stress_total / n_articles
    
    # ── Sentiment score: mean compound × 100 → range [-100, 100] ──
    sentiment_score = np.mean(sentiment_compounds) * 100
    
    # ── Environment score from news ──
    env_score = compute_environment_score_from_articles(df)
    
    return {
        'Sentiment_Score': round(float(sentiment_score), 2),
        'Crime_Score': round(float(crime_score), 2),
        'Sanitation_Score': round(float(sanitation_score), 2),
        'Social_Stress_Score': round(float(stress_score), 2),
        'Environment_Score': round(float(env_score) if env_score is not None else 3.0, 2),
    }


def integrate_review_sentiment(nlp_amenity_csv):
    """
    Also compute an overall sentiment from reviews (already done in step 1,
    but we integrate it here for the Sentiment_Score).
    """
    # This is handled by the Sentiment_Score from news for now.
    # The review-based sentiment is captured in the category scores.
    pass


def process_all_places():
    """
    Process all news articles and compute contextual scores.
    """
    print("=" * 60)
    print("  STEP 2: NLP-Based Contextual / Social Scoring")
    print("=" * 60)
    
    # Find all news files
    news_files = glob.glob(os.path.join(config.NEWS_DIR, "*_news.csv"))
    
    if not news_files:
        print(f"❌ No news files found in {config.NEWS_DIR}")
        return pd.DataFrame()
    
    # Also get all known place names
    all_places = set()
    if os.path.exists(config.DATA_DIR):
        all_places.update(d for d in os.listdir(config.DATA_DIR)
                         if os.path.isdir(os.path.join(config.DATA_DIR, d)))
    
    print(f"\nFound {len(news_files)} news files.\n")
    
    results = []
    
    for nf in sorted(news_files):
        basename = os.path.basename(nf)
        # Extract place name from filename: "dharavi_news.csv" → "dharavi"
        place_name = basename.replace('_news.csv', '')
        
        # Skip the combined "all_news" files
        if place_name in ('all', 'all_news_synthetic'):
            continue
        
        scores = process_news_for_place(nf)
        
        if scores:
            scores['place_name'] = place_name
            results.append(scores)
            status = "OK"
        else:
            # Default scores for places with no parseable news
            results.append({
                'place_name': place_name,
                'Sentiment_Score': 0.0,
                'Crime_Score': 0.0,
                'Sanitation_Score': 0.0,
                'Social_Stress_Score': 0.0,
                'Environment_Score': 3.0,
            })
            status = "default"
        
        print(f"  [{status:>7}] {place_name}")
    
    # ── Save output ──
    df_out = pd.DataFrame(results)
    cols = ['place_name'] + config.CONTEXT_SCORE_COLUMNS
    # Ensure all columns exist
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = 0.0
    df_out = df_out[cols]
    
    df_out.to_csv(config.NLP_CONTEXT_SCORES_CSV, index=False)
    
    print(f"\n✅ NLP context scores saved to: {config.NLP_CONTEXT_SCORES_CSV}")
    print(f"   Total places: {len(df_out)}")
    print(f"   Score ranges:")
    for col in [c for c in df_out.columns if c != 'place_name']:
        vals = df_out[col]
        print(f"     {col:>25}: min={vals.min():.2f}, mean={vals.mean():.2f}, max={vals.max():.2f}")
    
    return df_out


if __name__ == "__main__":
    process_all_places()
