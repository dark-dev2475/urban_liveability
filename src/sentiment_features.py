import pandas as pd
import os
import glob
import csv

# Define the root directory for our reviews
REVIEWS_DIR = 'reviews'
OUTPUT_FILE = 'sentiment_features.csv'

def calculate_sentiment_score(filepath):
    """
    Reads a single review CSV and calculates its score.
    Score = (good reviews) - (bad reviews)
    """
    try:
        df = pd.read_csv(filepath)
        
        # Ensure 'rating' column exists
        if 'rating' not in df.columns:
            print(f"  [Warning] 'rating' column not in {filepath}")
            return 0
            
        # We only saved 4-5 star and 1-2 star reviews
        good_reviews = len(df[df['rating'] >= 4])
        bad_reviews = len(df[df['rating'] <= 2])
        
        return good_reviews - bad_reviews
        
    except pd.errors.EmptyDataError:
        # This is normal, means the file was created but no reviews were found
        return 0
    except Exception as e:
        print(f"  [Error] Could not read {filepath}: {e}")
        return 0

def main():
    print("--- Building Stage 2: Amenity Quality (Sentiment) Features ---")
    
    # Get all place directories (e.g., 'dharavi', 'Andheri')
    try:
        all_places = [d for d in os.listdir(REVIEWS_DIR) if os.path.isdir(os.path.join(REVIEWS_DIR, d))]
    except FileNotFoundError:
        print(f"🚨 Error: Directory not found: '{REVIEWS_DIR}'")
        print("Please make sure you have run the review collection script first.")
        return

    if not all_places:
        print("🚨 Error: No place folders found in 'reviews/' directory.")
        return

    final_scores = []

    # Loop 1: Iterate through each PLACE
    for place_name in all_places:
        place_path = os.path.join(REVIEWS_DIR, place_name)
        print(f"\nProcessing place: {place_name}")
        
        # Find all review CSVs in this place's folder
        review_files = glob.glob(os.path.join(place_path, '*.csv'))
        
        if not review_files:
            print(f"  -> No review files found for {place_name}.")
            continue
            
        total_place_score = 0
        
        # Loop 2: Iterate through each CSV for that place
        for review_file in review_files:
            # Get the score for this one file (e.g., "Education")
            file_score = calculate_sentiment_score(review_file)
            
            # Add it to the place's total score
            total_place_score += file_score
            
        print(f"  -> Total Sentiment_Score: {total_place_score}")
        final_scores.append({
            'place_name': place_name,
            'Sentiment_Score': total_place_score
        })

    # --- Save all results to one master CSV ---
    if not final_scores:
        print("No scores were calculated.")
        return

    try:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['place_name', 'Sentiment_Score'])
            writer.writeheader()
            writer.writerows(final_scores)
            
        print(f"\n✅✅✅ Success! ✅✅✅")
        print(f"Feature engineering complete. All scores saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"🚨 Error: Could not write to output file {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    main()