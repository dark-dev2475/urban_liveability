import pandas as pd
import os
import glob
import csv
import re

# --- 1. DEFINE KEYWORDS & WEIGHTS ---

CRIME_KEYWORDS = {
    # High Severity (Weight: 10)
    'murder': 10, 'homicide': 10, 'killed': 10,
    # Medium-High Severity (Weight: 7)
    'assault': 7, 'stabbed': 7, 'rape': 7, 'violent': 7, 'molestation': 7,
    # Medium Severity (Weight: 5)
    'extortion': 5, 'robbery': 5, 'crime': 5, 'attack': 5,
    # Low Severity (Weight: 3)
    'theft': 3, 'burglary': 3, 'arrested': 3, 'drugs': 3, 'seized': 3, 'police': 3
}

SANITATION_KEYWORDS = {
    # High Severity (Weight: 5)
    'contamination': 5, 'contaminated': 5, 'sewage': 5, 'gutter': 5, 
    'no water': 5, 'water scarcity': 5, 'epidemic': 5,
    # Medium Severity (Weight: 3)
    'sanitation': 3, 'waste': 3, 'garbage': 3, 'trash': 3, 'dirty': 3,
    'unhygienic': 3, 'sewer': 3, 'toilets': 3,
    # Low Severity (Weight: 1)
    'cleaning': 1, 'health': 1, 'pollution': 1
}

# --- ⬇️ NEW FEATURE ⬇️ ---
# This captures "liveability" and "slum-like" *symptoms*
# WITHOUT using the target word "slum".
SOCIAL_STRESS_KEYWORDS = {
    # High Severity (Weight: 5)
    'unsafe': 5, 'dangerous': 5, 'poverty': 5, 'struggle': 5, 
    'no electricity': 5, 'eviction': 5,
    
    # Medium Severity (Weight: 3)
    'protest': 3, 'crowded': 3, 'overcrowded': 3, 'poor quality': 3,
    'hardship': 3, 'inadequate': 3
}
# --- ⬆️ END NEW FEATURE ⬆️ ---


# --- 2. SCRIPT CONFIGURATION ---
NEWS_DIR = 'news_articles'
OUTPUT_FILE = 'context_features.csv'

def calculate_scores_for_place(filepath):
    """
    Reads a single news CSV and calculates the total
    Crime, Sanitation, and Social Stress scores.
    """
    
    total_crime_score = 0
    total_sanitation_score = 0
    total_social_score = 0 # <-- New
    
    try:
        df = pd.read_csv(filepath)
        df['full_text'] = df['title'].fillna('') + ' ' + df['snippet'].fillna('')
        
        for text in df['full_text']:
            
            # --- Calculate Crime Score ---
            for keyword, weight in CRIME_KEYWORDS.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    total_crime_score += weight
            
            # --- Calculate Sanitation Score ---
            for keyword, weight in SANITATION_KEYWORDS.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    total_sanitation_score += weight

            # --- Calculate Social Stress Score ---
            for keyword, weight in SOCIAL_STRESS_KEYWORDS.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    total_social_score += weight

        return total_crime_score, total_sanitation_score, total_social_score # <-- New

    except pd.errors.EmptyDataError:
        return 0, 0, 0
    except Exception as e:
        print(f"  [Error] Could not read {filepath}: {e}")
        return 0, 0, 0

def main():
    print("--- Building Stage 3: Context (Crime, Sanitation, Social) Features ---")
    
    try:
        all_news_files = glob.glob(os.path.join(NEWS_DIR, '*_news.csv'))
    except FileNotFoundError:
        print(f"🚨 Error: Directory not found: '{NEWS_DIR}'")
        return

    if not all_news_files:
        print(f"🚨 Error: No '_news.csv' files found in '{NEWS_DIR}'")
        return

    final_scores = []

    for news_file in all_news_files:
        base_name = os.path.basename(news_file)
        place_name = base_name.replace('_news.csv', '')
        
        print(f"\nProcessing place: {place_name}")
        
        # Calculate all three scores
        crime_score, sanitation_score, social_score = calculate_scores_for_place(news_file) # <-- New
        
        print(f"  -> Crime_Score: {crime_score}")
        print(f"  -> Sanitation_Score: {sanitation_score}")
        print(f"  -> Social_Stress_Score: {social_score}") # <-- New
        
        final_scores.append({
            'place_name': place_name,
            'Crime_Score': crime_score,
            'Sanitation_Score': sanitation_score,
            'Social_Stress_Score': social_score # <-- New
        })

    if not final_scores:
        print("No scores were calculated.")
        return

    try:
        # Update fieldnames
        fieldnames = ['place_name', 'Crime_Score', 'Sanitation_Score', 'Social_Stress_Score']
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_scores)
            
        print(f"\n✅✅✅ Success! ✅✅✅")
        print(f"Feature engineering complete. All scores saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"🚨 Error: Could not write to output file {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    main()