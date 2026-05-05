"""
Urban Liveability Project — Central Configuration
===================================================
All paths, feature columns, weights, and keyword dictionaries.
"""
import os

# ─── Project Paths ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REVIEWS_DIR = os.path.join(PROJECT_ROOT, "reviews")
NEWS_DIR = os.path.join(PROJECT_ROOT, "news_articles")
SLUM_DETECTION_DIR = os.path.join(PROJECT_ROOT, "slum_detection")
PIPELINE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pipeline_output")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Ensure output dirs exist
for d in [PIPELINE_OUTPUT_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Input CSV Paths ─────────────────────────────────────────────────
LABELS_CSV = os.path.join(PROJECT_ROOT, "labels.csv")
MASTER_REFERENCE_CSV = os.path.join(PROJECT_ROOT, "dataset_comparison", "MASTER_REFERENCE.csv")
FUZZY_SCORES_CSV = os.path.join(PROJECT_ROOT, "fuzzy_scores.csv")
PLACE_NAMES_CSV = os.path.join(PROJECT_ROOT, "place_names_list.csv")

# ─── Pipeline Output Paths ───────────────────────────────────────────
NLP_AMENITY_SCORES_CSV = os.path.join(PIPELINE_OUTPUT_DIR, "nlp_amenity_scores.csv")
NLP_CONTEXT_SCORES_CSV = os.path.join(PIPELINE_OUTPUT_DIR, "nlp_context_scores.csv")
SATELLITE_SCORES_CSV = os.path.join(PIPELINE_OUTPUT_DIR, "satellite_scores.csv")
MASTER_DATASET_CSV = os.path.join(PIPELINE_OUTPUT_DIR, "master_dataset.csv")
AUGMENTED_DATASET_CSV = os.path.join(PIPELINE_OUTPUT_DIR, "augmented_dataset.csv")
FINAL_TRAINING_CSV = os.path.join(PIPELINE_OUTPUT_DIR, "final_training_data.csv")

# ─── Feature Columns ─────────────────────────────────────────────────
AMENITY_CATEGORIES = [
    "Education", "Environment", "Food", "Health",
    "Recreation", "Religion", "Services", "Transport"
]

AMENITY_SCORE_COLUMNS = [f"{cat}_Score" for cat in AMENITY_CATEGORIES]

CONTEXT_SCORE_COLUMNS = [
    "Sentiment_Score", "Crime_Score", "Sanitation_Score", "Social_Stress_Score", "Environment_Score"
]

SATELLITE_SCORE_COLUMN = ["Satellite_Score"]

ALL_FEATURE_COLUMNS = list(dict.fromkeys(AMENITY_SCORE_COLUMNS + CONTEXT_SCORE_COLUMNS + SATELLITE_SCORE_COLUMN))
# Note: Fuzzy_Score is intentionally EXCLUDED (redundant composite)

TARGET_COLUMN = "is_slum"

# ─── Review Category Mapping ─────────────────────────────────────────
# Maps review file name keywords → our standard category names
REVIEW_FILE_CATEGORY_MAP = {
    "education": "Education",
    "food": "Food",
    "drink": "Food",
    "health": "Health",
    "recreation": "Recreation",
    "leisure": "Recreation",
    "religion": "Religion",
    "temple": "Religion",
    "shopping": "Services",
    "services": "Services",
    "service": "Services",
    "safety": "Services",
    "transport": "Transport",
    "environment": "Environment",
    "waste": "Environment",
}

# ─── Amenity Data Category Mapping ───────────────────────────────────
AMENITY_FILE_CATEGORY_MAP = {
    "education": "Education",
    "school": "Education",
    "college": "Education",
    "food": "Food",
    "drink": "Food",
    "restaurant": "Food",
    "health": "Health",
    "hospital": "Health",
    "clinic": "Health",
    "recreation": "Recreation",
    "leisure": "Recreation",
    "park": "Recreation",
    "religion": "Religion",
    "temple": "Religion",
    "shopping": "Services",
    "services": "Services",
    "service": "Services",
    "transport": "Transport",
    "mobility": "Transport",
    "bus": "Transport",
    "train": "Transport",
    "environment": "Environment",
    "pollution": "Environment",
    "waste": "Environment",
    "green": "Environment",
    "air": "Environment",
}

# ─── NLP Amenity Scoring Weights ─────────────────────────────────────
NLP_WEIGHT_SEMANTIC = 0.50   # VADER sentiment weight
NLP_WEIGHT_RATING = 0.30    # Star rating weight
NLP_WEIGHT_VOLUME = 0.20    # Review volume weight
NLP_VOLUME_CAP = 2.0        # Max bonus from volume
NLP_DIVERSITY_CAP = 1.5     # Max multiplier from amenity diversity

# ─── Crime Keyword Dictionary (keyword → severity weight) ────────────
CRIME_KEYWORDS = {
    # High severity (10)
    'murder': 10, 'homicide': 10, 'killed': 10, 'shooting': 10,
    # Medium-high severity (7)
    'assault': 7, 'stabbed': 7, 'rape': 7, 'violent': 7, 'molestation': 7,
    'kidnapping': 7, 'gang': 7,
    # Medium severity (5)
    'extortion': 5, 'robbery': 5, 'crime': 5, 'attack': 5, 'arson': 5,
    # Low severity (3)
    'theft': 3, 'burglary': 3, 'arrested': 3, 'drugs': 3, 'seized': 3,
    'police': 3, 'vandalism': 3,
}

# ─── Sanitation Keyword Dictionary ────────────────────────────────────
SANITATION_KEYWORDS = {
    # High severity (5)
    'contamination': 5, 'contaminated': 5, 'sewage': 5, 'gutter': 5,
    'no water': 5, 'water scarcity': 5, 'epidemic': 5, 'cholera': 5,
    'dengue': 5, 'malaria': 5,
    # Medium severity (3)
    'sanitation': 3, 'waste': 3, 'garbage': 3, 'trash': 3, 'dirty': 3,
    'unhygienic': 3, 'sewer': 3, 'toilets': 3, 'drain': 3, 'stench': 3,
    # Low severity (1)
    'cleaning': 1, 'pollution': 1, 'disposal': 1,
}

# ─── Social Stress Keyword Dictionary ─────────────────────────────────
SOCIAL_STRESS_KEYWORDS = {
    # High severity (5)
    'unsafe': 5, 'dangerous': 5, 'poverty': 5, 'struggle': 5,
    'no electricity': 5, 'eviction': 5, 'demolition': 5,
    'displacement': 5, 'homeless': 5,
    # Medium severity (3)
    'protest': 3, 'crowded': 3, 'overcrowded': 3, 'poor quality': 3,
    'hardship': 3, 'inadequate': 3, 'encroachment': 3,
    'illegal': 3, 'unplanned': 3,
}

# ─── Environment Keyword Dictionary ───────────────────────────────────
ENVIRONMENT_KEYWORDS_NEGATIVE = {
    'pollution': 5, 'polluted': 5, 'toxic': 5, 'smog': 5,
    'deforestation': 4, 'industrial waste': 4, 'chemical': 4,
    'noise pollution': 3, 'dust': 3, 'smoke': 3, 'fumes': 3,
    'dumping': 3, 'landfill': 3, 'hazardous': 4,
}

ENVIRONMENT_KEYWORDS_POSITIVE = {
    'green': 3, 'park': 3, 'garden': 3, 'tree': 2, 'plantation': 3,
    'clean air': 4, 'eco-friendly': 3, 'sustainable': 3,
    'nature': 2, 'biodiversity': 3, 'lake': 2, 'river': 2,
}

# ─── Model Training Config ────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 10
SYNTHETIC_TARGET_TOTAL = 500   # Target total samples after augmentation
