"""
Urban Liveability Assessment Model
----------------------------------
A modular fuzzy logic system for evaluating urban liveability based on 
geographic accessibility to various urban amenities and services.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import warnings
warnings.filterwarnings('ignore')

# Define paths
<<<<<<< HEAD
DATA_DIR = r"c:\Users\gagan\OneDrive\Desktop\urban_bot\data\dharavi"
=======
DATA_DIR = r"c:\Users\gagan\OneDrive\Desktop\urban_bot\data\jhalwa"
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a
OUTPUT_DIR = r"c:\Users\gagan\OneDrive\Desktop\urban_bot\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Category weights (determined through literature review and expert input)
CATEGORY_WEIGHTS = {
<<<<<<< HEAD
    # Institutional (0.25 total)
    'services': 0.25,

    # Social (0.25 total)
    'healthcare': 0.10,
    'education': 0.10,
    'recreation': 0.03,
    'food': 0.02,

    # Economic (0.05 total)
    'economic_services': 0.05,

    # Physical (0.45 total)
    'transport': 0.20,
    'environment': 0.25
}


=======
    'transport': 0.25,
    'healthcare': 0.20,
    'education': 0.20,
    'services': 0.15,
    'recreation': 0.10,
    'food': 0.10
}

>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a
# Regional context parameters
REGIONAL_CONTEXT = {
    'urban_density': 'medium',  # Options: 'low', 'medium', 'high'
    'region_type': 'developing',  # Options: 'developing', 'developed'
    'population': 'medium',  # Options: 'low', 'medium', 'high'
    'max_reasonable_distance': 5.0  # Maximum distance (km) that's considered
}

# =============================================================================
# DATA LOADING AND PROCESSING MODULE
<<<<<<< HEAD
def load_location_data(location_name, data_path):
    """
    Fully generalized loader.
    This version loads all .csv files from the directory and
    matches them by category keywords, regardless of the filename prefix.
    """

    print(f"\n🔍 Loading data for area: {location_name} from {data_path}")

    category_keywords = {
        "Transport": ["transport", "mobility", "bus", "train"],
        "Health": ["health", "hospital", "clinic"],
        "Education": ["education", "school", "college"],
        "Services": ["service", "shopping"],
        "Recreation": ["recreation", "park", "leisure"],
        "Food": ["food", "drink", "restaurant"],
        "Environment": ["environment", "pollution", "air", "green","waste_management"],
        "Religion": ["religion", "temple"]
    }

    # Start with empty DataFrames
    data = {cat: pd.DataFrame() for cat in category_keywords.keys()}

    try:
        # list all CSV files
        files = [f for f in os.listdir(data_path) if f.lower().endswith(".csv")]
    except FileNotFoundError:
        print(f"❌ No folder found at path: {data_path}")
        return data

    if not files:
        print(f"❌ No CSV files found in {data_path}")
        return data

    for file in files:
        f_lower = file.lower()

        # --- THIS IS THE FIX ---
        # The bad "if location_name.lower() not in f_lower:" check
        # has been completely REMOVED.
        
        matched_category = None
        for category, keywords in category_keywords.items():
            # Check if any keyword (e.g., "health", "education")
            # is in the filename.
            if any(kw in f_lower for kw in keywords):
                matched_category = category
                break

        if matched_category:
            try:
                df = pd.read_csv(os.path.join(data_path, file))
                data[matched_category] = df
                print(f"📥 Loaded {matched_category:<12} → {file}")
            except Exception as e:
                print(f"⚠ Error reading {file}: {e}")
        else:
            # This will tell you which files are failing
            print(f"   [Skipping] Could not match a category for file: {file}")

    return data

    
    # Alternative mappings in case file naming is different
    alternative_mappings = {
    "Transport": ["transport", "mobility", "road", "traffic"],
    "Health": ["health", "hospital", "clinic", "medical"],
    "Education": ["education", "school", "college"],
    "Services": ["services", "shopping", "store", "retail"],
    "Recreation": ["recreation", "park", "leisure"],
    "Food": ["food", "restaurant", "cafe", "eatery"],
    "Religion": ["religion", "temple", "mosque", "church"],
    "Environment": ["environment", "pollution", "air", "green"]
}

    
    
=======
# =============================================================================

def load_category_data(category):
    """Load data for a specific category from CSV file"""
    file_path = os.path.join(DATA_DIR, f"Jhalwa_Prayagraj_{category}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records for {category}")
        return df
    else:
        print(f"Warning: Data file for {category} not found at {file_path}")
        return pd.DataFrame()

def load_all_data():
    """Load all available data categories"""
    categories = [
        "Transport", 
        "Health", 
        "Education", 
        "Shopping & Services", 
        "Recreation & Leisure", 
        "Food & Drink", 
        "Religion"  # Added religion as it can be important in Indian context
    ]
    
    data = {}
    for category in categories:
        data[category] = load_category_data(category)
    
    return data

def load_location_data(location_name, data_path=None):
    """
    Load data for a specific location, mapping file names to category names
    
    Args:
        location_name: Name of the location
        data_path: Path to data files (if None, uses default path)
    
    Returns:
        Dictionary with dataframes for each category
    """
    if data_path is None:
        data_path = DATA_DIR
    
    # Define mapping of file names to category keys
    file_to_category = {
        f"{location_name}_Prayagraj_Transport.csv": "Transport",
        f"{location_name}_Prayagraj_Health.csv": "Health",
        f"{location_name}_Prayagraj_Education.csv": "Education",
        f"{location_name}_Prayagraj_Shopping & Services.csv": "Services",
        f"{location_name}_Prayagraj_Recreation & Leisure.csv": "Recreation",
        f"{location_name}_Prayagraj_Food & Drink.csv": "Food",
        f"{location_name}_Prayagraj_Religion.csv": "Religion",
        f"{location_name}_Prayagraj_Environment.csv": "Environment"  # Added Environment category
    }
    
    # Alternative mappings in case file naming is different
    alternative_mappings = {
        "Transport": ["Transport", "Transportation", "transit"],
        "Health": ["Health", "Healthcare", "Medical"],
        "Education": ["Education", "Educational", "Schools"],
        "Services": ["Services", "Shopping & Services", "Shopping"],
        "Recreation": ["Recreation", "Recreation & Leisure", "Leisure", "Parks"],
        "Food": ["Food", "Food & Drink", "Restaurants"],
        "Environment": ["Environment", "Pollution", "Air Quality", "Environmental"]  # Added Environment alternatives
    }
    
    data = {}
    
    # Check if we need to list directory to find files
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} does not exist")
        return data
    
    # First try with direct mapping
    for filename, category in file_to_category.items():
        file_path = os.path.join(data_path, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} records for {category}")
            data[category] = df
    
    # If we're missing categories, try to find files with similar names
    if len(data) < len(alternative_mappings):
        for file in os.listdir(data_path):
            if file.endswith('.csv'):
                # Skip files we've already loaded
                already_loaded = False
                for filename in file_to_category.keys():
                    if filename.lower() == file.lower():
                        already_loaded = True
                        break
                
                if already_loaded:
                    continue
                
                # Try to match with alternative names
                file_path = os.path.join(data_path, file)
                df = pd.read_csv(file_path)
                
                # Try to determine category from filename
                matched = False
                for category, alternatives in alternative_mappings.items():
                    if category not in data:  # Only if not already loaded
                        for alt in alternatives:
                            if alt.lower() in file.lower():
                                print(f"Loaded {len(df)} records for {category} from {file}")
                                data[category] = df
                                matched = True
                                break
                        if matched:
                            break
    
    return data
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a

def analyze_distance_distributions(all_data):
    """Analyze the distribution of distances for each category"""
    distance_stats = {}
    
    for category, df in all_data.items():
<<<<<<< HEAD
        if df is None or df.empty or 'Distance_km' not in df.columns:
            continue
        
        dist = df['Distance_km'].dropna()
        if dist.empty:
            continue

        stats = {
            'min': dist.min(),
            'max': dist.max(),
            'mean': dist.mean(),
            'median': dist.median(),
            'std': dist.std(),
            'count': len(dist),
            'percentiles': {
                '10%': dist.quantile(0.1),
                '25%': dist.quantile(0.25),
                '50%': dist.quantile(0.5),
                '75%': dist.quantile(0.75),
                '90%': dist.quantile(0.9)
            }
        }

=======
        if df.empty or 'Distance_km' not in df.columns:
            continue
        
        stats = {
            'min': df['Distance_km'].min(),
            'max': df['Distance_km'].max(),
            'mean': df['Distance_km'].mean(),
            'median': df['Distance_km'].median(),
            'std': df['Distance_km'].std(),
            'count': len(df),
            'percentiles': {
                '10%': df['Distance_km'].quantile(0.1),
                '25%': df['Distance_km'].quantile(0.25),
                '50%': df['Distance_km'].quantile(0.5),
                '75%': df['Distance_km'].quantile(0.75),
                '90%': df['Distance_km'].quantile(0.9)
            }
        }
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a
        distance_stats[category] = stats
        
        print(f"\n{category} Distance Statistics:")
        print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f} km")
        print(f"  Mean: {stats['mean']:.2f} km, Median: {stats['median']:.2f} km")
        print(f"  Standard Deviation: {stats['std']:.2f} km")
<<<<<<< HEAD
        print(f"  Percentiles: "
              f"25%={stats['percentiles']['25%']:.2f}, "
              f"50%={stats['percentiles']['50%']:.2f}, "
              f"75%={stats['percentiles']['75%']:.2f}")
    
    return distance_stats


# =============================================================================
# GEOCODING AND DISTANCE CALCULATION MODULE
# =============================================================================  
=======
        print(f"  Percentiles: 25%={stats['percentiles']['25%']:.2f}, 50%={stats['percentiles']['50%']:.2f}, 75%={stats['percentiles']['75%']:.2f}")
    
    return distance_stats

# =============================================================================
# GEOCODING AND DISTANCE CALCULATION MODULE
# =============================================================================
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def calculate_distances_from_point(user_lat, user_lon, amenities_df):
    """
    Calculate distances from a specific point to all amenities in the dataframe
<<<<<<< HEAD
    """
    if amenities_df.empty:
        return amenities_df

    if 'Latitude' not in amenities_df.columns or 'Longitude' not in amenities_df.columns:
        print("Warning: Latitude/Longitude columns not found. Using existing Distance_km if available.")
        return amenities_df

    # Convert coordinates to radians
    lat1 = np.radians(user_lat)
    lon1 = np.radians(user_lon)
    lat2 = np.radians(amenities_df['Latitude'].values)
    lon2 = np.radians(amenities_df['Longitude'].values)

    # Haversine formula (vectorized)
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.minimum(1, np.sqrt(a)))
    r = 6371

    amenities_df = amenities_df.copy()
    amenities_df['Distance_km'] = r * c
    return amenities_df


def determine_area_from_coordinates(lat, lon):
    """
    Dynamically determine the area based on available dataset files.
    This uses the prefix of filenames inside DATA_DIR to identify the area.
    
    Example:
        CivilLines_Transport.csv → area = 'CivilLines'
        Jhalwa_Health.csv → area = 'Jhalwa'
    
    The nearest area center is chosen.
    """

    area_centers = {}  # {area_name: (lat, lon)}

    # Scan DATA_DIR for files
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv") and "_" in file:
            area_name = file.split("_")[0]  # prefix before '_'
            
            # Load ANY file for that area to fetch coordinates
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            
            if "Latitude" in df.columns and "Longitude" in df.columns:
                # Use mean center of coordinates for that area
                mean_lat = df["Latitude"].mean()
                mean_lon = df["Longitude"].mean()
                area_centers[area_name] = (mean_lat, mean_lon)

    if not area_centers:
        raise ValueError("No area information detected from dataset files.")

    # Compute distance to each area's center
    min_distance = float("inf")
    closest_area = None

    for area, (a_lat, a_lon) in area_centers.items():
        dist = haversine_distance(lat, lon, a_lat, a_lon)
        if dist < min_distance:
            min_distance = dist
            closest_area = area

    print(f"Detected Area: {closest_area} (distance = {min_distance:.2f} km)")
    return closest_area

=======
    
    Args:
        user_lat: User's latitude
        user_lon: User's longitude
        amenities_df: DataFrame with amenity locations (must have 'Latitude' and 'Longitude' columns)
    
    Returns:
        DataFrame with added 'Distance_km' column calculated from the user's location
    """
    if amenities_df.empty:
        return amenities_df
    
    # Check if coordinate columns exist
    if 'Latitude' not in amenities_df.columns or 'Longitude' not in amenities_df.columns:
        print("Warning: Latitude/Longitude columns not found. Using existing Distance_km if available.")
        return amenities_df
    
    # Calculate distances using Haversine formula
    amenities_df = amenities_df.copy()
    amenities_df['Distance_km'] = amenities_df.apply(
        lambda row: haversine_distance(user_lat, user_lon, row['Latitude'], row['Longitude']),
        axis=1
    )
    
    return amenities_df

def determine_area_from_coordinates(lat, lon):
    """
    Determine which area/neighborhood the coordinates fall into
    This is a simple implementation - in real applications, you'd use proper GIS data
    
    For now, we'll assume coordinates in Prayagraj district map to Jhalwa if they're close
    """
    # Approximate Jhalwa area boundaries (these would need to be more precise in real implementation)
    jhalwa_bounds = {
        'min_lat': 25.40,
        'max_lat': 25.50,
        'min_lon': 81.80,
        'max_lon': 81.90
    }
    
    if (jhalwa_bounds['min_lat'] <= lat <= jhalwa_bounds['max_lat'] and 
        jhalwa_bounds['min_lon'] <= lon <= jhalwa_bounds['max_lon']):
        return "Jhalwa"
    else:
        # For now, default to Jhalwa since that's our available dataset
        print(f"Coordinates ({lat:.4f}, {lon:.4f}) outside Jhalwa bounds, but using Jhalwa data as reference")
        return "Jhalwa"
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a

# =============================================================================
# ACCESSIBILITY CALCULATION MODULE
# =============================================================================

def calculate_air_quality_score(df):
    """
<<<<<<< HEAD
    Calculate air quality score (0–10 scale) from environmental pollutant data.
    """

    if df.empty or 'Pollutant' not in df.columns or 'Value' not in df.columns:
        print("No valid environmental data, returning minimum score")
        return 0.0

    # Normalize pollutant names for reliability
    df = df.copy()
    df['Pollutant'] = df['Pollutant'].astype(str).str.strip().str.upper()

    # WHO guideline values (uppercased keys)
    who_guidelines = {
        'PM2.5': 5,
        'PM10': 15,
        'NO2': 10,
        'SO2': 40,
        'O3': 100
    }

    pollutant_scores = []

    for pollutant, guideline in who_guidelines.items():
        pollutant_data = df[df['Pollutant'] == pollutant]

        if not pollutant_data.empty:
            avg_value = pollutant_data['Value'].mean()

            if avg_value <= guideline:
                score = 10.0
            else:
                ratio = avg_value / guideline

                # Prevent overflow
                ratio = min(ratio, 20)

                score = 10.0 * np.exp(-0.35 * (ratio - 1))
                score = max(0, min(score, 10))

=======
    Calculate air quality score from environmental data
    
    Args:
        df: DataFrame with environmental data including pollutant measurements
        
    Returns:
        Air quality score (0-10 scale, where 10 is best/cleanest air)
    """
    if df.empty or 'Pollutant' not in df.columns or 'Value' not in df.columns:
        print("No valid environmental data, returning minimum score")
        return 0.0
    
    # WHO guideline values for common pollutants (annual mean)
    who_guidelines = {
        'PM2.5': 5,    # µg/m³
        'PM10': 15,    # µg/m³
        'NO2': 10,     # µg/m³
        'SO2': 40,     # µg/m³
        'O3': 100      # µg/m³, 8-hour mean
    }
    
    # Calculate normalized scores for each pollutant (0-10 scale, higher is better)
    pollutant_scores = []
    
    for pollutant, guideline in who_guidelines.items():
        # Filter data for this pollutant
        pollutant_data = df[df['Pollutant'] == pollutant]
        
        if not pollutant_data.empty:
            # Get average value
            avg_value = pollutant_data['Value'].mean()
            
            # Calculate score based on ratio to WHO guideline
            # Perfect score (10) if at or below guideline
            # Score decreases as concentration increases above guideline
            if avg_value <= guideline:
                score = 10.0
            else:
                # Score decreases as pollution increases (inverse relationship)
                # Using a function that gives 5.0 at 2x guideline and approaches 0 asymptotically
                ratio = avg_value / guideline
                score = 10.0 * np.exp(-0.35 * (ratio - 1))
                score = max(0, min(score, 10))  # Clamp between 0 and 10
            
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a
            pollutant_scores.append({
                'pollutant': pollutant,
                'value': avg_value,
                'guideline': guideline,
                'score': score
            })
<<<<<<< HEAD

    if not pollutant_scores:
        return 0.0

=======
    
    if not pollutant_scores:
        return 0.0
    
    # Calculate overall air quality score - weighted average
    # PM2.5 and PM10 are given higher weights as they're most health-relevant
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a
    weights = {
        'PM2.5': 0.35,
        'PM10': 0.25,
        'NO2': 0.15,
        'SO2': 0.10,
        'O3': 0.15
    }
<<<<<<< HEAD

    total_weight = 0
    weighted_score = 0

    for ps in pollutant_scores:
        pollutant = ps['pollutant']
        weight = weights.get(pollutant, 0.1)
        weighted_score += ps['score'] * weight
        total_weight += weight
        print(f"  {pollutant}: {ps['value']:.1f} µg/m³ (WHO: {ps['guideline']}) → Score: {ps['score']:.1f}/10")

    final_score = weighted_score / total_weight
    print(f"Overall air quality score: {final_score:.1f}/10")

    return final_score

=======
    
    total_weight = 0
    weighted_score = 0
    
    for ps in pollutant_scores:
        pollutant = ps['pollutant']
        weight = weights.get(pollutant, 0.1)  # Default weight if not in the dict
        weighted_score += ps['score'] * weight
        total_weight += weight
        print(f"  {pollutant}: {ps['value']:.1f} µg/m³ (WHO: {ps['guideline']}) → Score: {ps['score']:.1f}/10")
    
    if total_weight > 0:
        final_score = weighted_score / total_weight
        print(f"Overall air quality score: {final_score:.1f}/10")
        return final_score
    else:
        return 0.0
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a

def calculate_category_score(df, category, regional_context=REGIONAL_CONTEXT):
    """
    Calculate accessibility score with improved methodology:
    1. Consider variety of amenities (not just distance)
    2. Apply distance decay function
    3. Account for regional context
    """
    # Handle environment category separately using air quality calculation
    if category == 'Environment':
        return calculate_air_quality_score(df)
    
    if df.empty or 'Distance_km' not in df.columns:
        print(f"No valid data for {category}, returning minimum score")
        return 0.0
    
    # Get contextual parameters
    max_distance = regional_context['max_reasonable_distance']
    
    # Apply regional context adjustments
    context_factor = 1.0
    if regional_context['urban_density'] == 'high':
        # In high-density areas, expect shorter distances
        context_factor = 0.8
    elif regional_context['urban_density'] == 'low':
        # In low-density areas, accept longer distances
        context_factor = 1.2
    
    # Apply distance decay function (closer amenities have exponentially more value)
    df = df.copy()
    df['distance_factor'] = df['Distance_km'].apply(
        lambda d: np.exp(-0.25 * d * context_factor) if d <= max_distance else 0
    )
    
    # Consider diversity of amenities (unique types have added value)
    if 'Amenity' in df.columns:
        unique_amenities = df['Amenity'].nunique()
        diversity_factor = min(1 + (unique_amenities / 10), 1.5)  # Cap at 1.5x bonus
    else:
        diversity_factor = 1.0
    
    # Calculate weighted scores based on amenity type
    if 'Amenity' in df.columns:
        # Define amenity-specific weights for each category
        amenity_weights = {}
        
        if category == 'Transport':
            amenity_weights = {
                'bus_stop': 1.2,
                'train_station': 1.5,
                'metro_station': 1.5,
                'taxi': 0.8,
                'bicycle_rental': 0.7,
                'parking': 0.5
            }
        elif category == 'Health':
            amenity_weights = {
                'hospital': 1.5,
                'clinic': 1.2,
                'pharmacy': 1.0,
                'doctors': 0.8
            }
        elif category == 'Education':
            amenity_weights = {
                'university': 1.5,
                'college': 1.3,
                'school': 1.2,
                'kindergarten': 1.0,
                'library': 0.8
            }
        
        # Apply weights where possible
        df['amenity_weight'] = df['Amenity'].apply(
            lambda a: amenity_weights.get(a, 1.0) if pd.notna(a) else 1.0
        )
    else:
        df['amenity_weight'] = 1.0
    
    # Calculate final accessibility value
    df['weighted_factor'] = df['distance_factor'] * df['amenity_weight']
    
    # Take top 5 amenities with highest weighted factors
    top_amenities = df.nlargest(5, 'weighted_factor')
    
    if len(top_amenities) > 0:
        # Calculate accessibility score (0-10 scale, 10 is best)
        access_score = min(top_amenities['weighted_factor'].sum() * diversity_factor, 10)
        return access_score
    else:
        return 0.0

def calculate_category_score_geocoded(df, category, user_lat, user_lon, regional_context=REGIONAL_CONTEXT):
    """
    Calculate accessibility score for a specific geocoded location with improved methodology:
    1. Calculate actual distances from user's coordinates to all amenities
    2. Consider variety of amenities (not just distance)
    3. Apply distance decay function
    4. Account for regional context
    """
    # Handle environment category separately using air quality calculation
    if category == 'Environment':
        return calculate_air_quality_score(df)
    
    if df.empty:
        print(f"No valid data for {category}, returning minimum score")
        return 0.0
    
    # Calculate distances from user's location to all amenities
    df_with_distances = calculate_distances_from_point(user_lat, user_lon, df)
    
    if 'Distance_km' not in df_with_distances.columns:
        print(f"Could not calculate distances for {category}, returning minimum score")
        return 0.0
    
    # Get contextual parameters
    max_distance = regional_context['max_reasonable_distance']
    
    # Apply regional context adjustments
    context_factor = 1.0
    if regional_context['urban_density'] == 'high':
        # In high-density areas, expect shorter distances
        context_factor = 0.8
    elif regional_context['urban_density'] == 'low':
        # In low-density areas, accept longer distances
        context_factor = 1.2
    
    # Apply distance decay function (closer amenities have exponentially more value)
    df_with_distances = df_with_distances.copy()
    df_with_distances['distance_factor'] = df_with_distances['Distance_km'].apply(
        lambda d: np.exp(-0.25 * d * context_factor) if d <= max_distance else 0
    )
    
    # Consider diversity of amenities (unique types have added value)
    if 'Amenity' in df_with_distances.columns:
        unique_amenities = df_with_distances['Amenity'].nunique()
        diversity_factor = min(1 + (unique_amenities / 10), 1.5)  # Cap at 1.5x bonus
    else:
        diversity_factor = 1.0
    
    # Calculate weighted scores based on amenity type
    if 'Amenity' in df_with_distances.columns:
        # Define amenity-specific weights for each category
        amenity_weights = {}
        
        if category == 'Transport':
            amenity_weights = {
                'bus_stop': 1.2,
                'train_station': 1.5,
                'metro_station': 1.5,
                'taxi': 0.8,
                'bicycle_rental': 0.7,
                'parking': 0.5
            }
        elif category == 'Health':
            amenity_weights = {
                'hospital': 1.5,
                'clinic': 1.2,
                'pharmacy': 1.0,
                'doctors': 0.8
            }
        elif category == 'Education':
            amenity_weights = {
                'university': 1.5,
                'college': 1.3,
                'school': 1.2,
                'kindergarten': 1.0,
                'library': 0.8
            }
        
        # Apply weights where possible
        df_with_distances['amenity_weight'] = df_with_distances['Amenity'].apply(
            lambda a: amenity_weights.get(a, 1.0) if pd.notna(a) else 1.0
        )
    else:
        df_with_distances['amenity_weight'] = 1.0
    
    # Calculate final accessibility value
    df_with_distances['weighted_factor'] = df_with_distances['distance_factor'] * df_with_distances['amenity_weight']
    
    # Take top 5 amenities with highest weighted factors
    top_amenities = df_with_distances.nlargest(5, 'weighted_factor')
    
    if len(top_amenities) > 0:
        # Calculate accessibility score (0-10 scale, 10 is best)
        access_score = min(top_amenities['weighted_factor'].sum() * diversity_factor, 10)
        
        # Print details for debugging
        print(f"{category} accessibility from ({user_lat:.4f}, {user_lon:.4f}):")
        print(f"  Closest amenities: {top_amenities['Distance_km'].min():.2f}km - {top_amenities['Distance_km'].max():.2f}km")
        print(f"  Accessibility score: {access_score:.2f}/10")
        
        return access_score
    else:
        return 0.0
<<<<<<< HEAD
   
=======
    """
    Calculate accessibility score with improved methodology:
    1. Consider variety of amenities (not just distance)
    2. Apply distance decay function
    3. Account for regional context
    """
    # Handle environment category separately using air quality calculation
    if category == 'Environment':
        return calculate_air_quality_score(df)
    
    if df.empty or 'Distance_km' not in df.columns:
        print(f"No valid data for {category}, returning minimum score")
        return 0.0
    
    # Get contextual parameters
    max_distance = regional_context['max_reasonable_distance']
    
    # Apply regional context adjustments
    context_factor = 1.0
    if regional_context['urban_density'] == 'high':
        # In high-density areas, expect shorter distances
        context_factor = 0.8
    elif regional_context['urban_density'] == 'low':
        # In low-density areas, accept longer distances
        context_factor = 1.2
    
    # Apply distance decay function (closer amenities have exponentially more value)
    df = df.copy()
    df['distance_factor'] = df['Distance_km'].apply(
        lambda d: np.exp(-0.25 * d * context_factor) if d <= max_distance else 0
    )
    
    # Consider diversity of amenities (unique types have added value)
    if 'Amenity' in df.columns:
        unique_amenities = df['Amenity'].nunique()
        diversity_factor = min(1 + (unique_amenities / 10), 1.5)  # Cap at 1.5x bonus
    else:
        diversity_factor = 1.0
    
    # Calculate weighted scores based on amenity type
    if 'Amenity' in df.columns:
        # Define amenity-specific weights for each category
        amenity_weights = {}
        
        if category == 'Transport':
            amenity_weights = {
                'bus_stop': 1.2,
                'train_station': 1.5,
                'metro_station': 1.5,
                'taxi': 0.8,
                'bicycle_rental': 0.7,
                'parking': 0.5
            }
        elif category == 'Health':
            amenity_weights = {
                'hospital': 1.5,
                'clinic': 1.2,
                'pharmacy': 1.0,
                'doctors': 0.8
            }
        elif category == 'Education':
            amenity_weights = {
                'university': 1.5,
                'college': 1.3,
                'school': 1.2,
                'kindergarten': 1.0,
                'library': 0.8
            }
        
        # Apply weights where possible
        df['amenity_weight'] = df['Amenity'].apply(
            lambda a: amenity_weights.get(a, 1.0) if pd.notna(a) else 1.0
        )
    else:
        df['amenity_weight'] = 1.0
    
    # Calculate final accessibility value
    df['weighted_factor'] = df['distance_factor'] * df['amenity_weight']
    
    # Take top 5 amenities with highest weighted factors
    top_amenities = df.nlargest(5, 'weighted_factor')
    
    if len(top_amenities) > 0:
        # Calculate accessibility score (0-10 scale, 10 is best)
        access_score = min(top_amenities['weighted_factor'].sum() * diversity_factor, 10)
        return access_score
    else:
        return 0.0
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a

# =============================================================================
# FUZZY LOGIC MODULE
# =============================================================================

def define_fuzzy_system(distance_stats=None):
    """Define an advanced fuzzy logic system for urban liveability"""
    
    # Define input variables (accessibility scores, 0-10 scale where 10 is best)
    transport_access = ctrl.Antecedent(np.arange(0, 11, 0.1), 'transport_access')
    healthcare_access = ctrl.Antecedent(np.arange(0, 11, 0.1), 'healthcare_access')
    education_access = ctrl.Antecedent(np.arange(0, 11, 0.1), 'education_access')
    services_access = ctrl.Antecedent(np.arange(0, 11, 0.1), 'services_access')
    recreation_access = ctrl.Antecedent(np.arange(0, 11, 0.1), 'recreation_access')
    food_access = ctrl.Antecedent(np.arange(0, 11, 0.1), 'food_access')
    env_quality = ctrl.Antecedent(np.arange(0, 11, 0.1), 'env_quality')  # New environmental quality input
    
    # Define output variable
    liveability = ctrl.Consequent(np.arange(0, 101, 1), 'liveability')
    
    # Define membership functions for inputs (higher is better)
    for input_var in [transport_access, healthcare_access, education_access, 
                      services_access, recreation_access, food_access, env_quality]:
        input_var['poor'] = fuzz.trapmf(input_var.universe, [0, 0, 2, 4])
        input_var['moderate'] = fuzz.trimf(input_var.universe, [2, 5, 8])
        input_var['good'] = fuzz.trapmf(input_var.universe, [6, 8, 10, 10])
    
    # Define more nuanced output membership functions
    liveability['very_low'] = fuzz.trapmf(liveability.universe, [0, 0, 20, 30])
    liveability['low'] = fuzz.trimf(liveability.universe, [20, 35, 50])
    liveability['medium'] = fuzz.trimf(liveability.universe, [40, 55, 70])
    liveability['high'] = fuzz.trimf(liveability.universe, [60, 75, 90])
    liveability['very_high'] = fuzz.trapmf(liveability.universe, [80, 90, 100, 100])
    
    # Define a comprehensive set of fuzzy rules
    rules = []
    
    # Transportation is fundamental - strong influence
    rules.append(ctrl.Rule(transport_access['good'], liveability['high']))
    rules.append(ctrl.Rule(transport_access['poor'], liveability['low']))
    
    # Healthcare and education are essential services
    rules.append(ctrl.Rule(healthcare_access['good'] & education_access['good'], liveability['high']))
    rules.append(ctrl.Rule(healthcare_access['poor'] & education_access['poor'], liveability['very_low']))
    
    # Environmental quality rules
    rules.append(ctrl.Rule(env_quality['good'], liveability['high']))
    rules.append(ctrl.Rule(env_quality['poor'], liveability['low']))
    
    # Combined accessibility rules
    rules.append(ctrl.Rule(
        transport_access['good'] & healthcare_access['good'] & education_access['good'], 
        liveability['very_high']
    ))
    
    # Environmental quality combined rules
    rules.append(ctrl.Rule(
        env_quality['good'] & healthcare_access['good'],
        liveability['high']  # Good environment and healthcare are important for health
    ))
    
    rules.append(ctrl.Rule(
        env_quality['poor'] & healthcare_access['good'],
        liveability['medium']  # Good healthcare partially compensates for poor environment
    ))
    
    rules.append(ctrl.Rule(
        env_quality['poor'] & healthcare_access['poor'],
        liveability['very_low']  # Both poor environment and healthcare is very bad
    ))
    
    # Moderate everything rule
    rules.append(ctrl.Rule(
        transport_access['moderate'] & healthcare_access['moderate'] & 
        education_access['moderate'] & services_access['moderate'] &
        recreation_access['moderate'] & food_access['moderate'] &
        env_quality['moderate'],
        liveability['medium']
    ))
    
    # Services, recreation, and food enhance liveability but aren't as critical
    rules.append(ctrl.Rule(services_access['good'] & food_access['good'], liveability['high']))
    rules.append(ctrl.Rule(recreation_access['good'], liveability['high']))
    
    # Advanced combination rules
    rules.append(ctrl.Rule(
        transport_access['good'] & healthcare_access['good'] & 
        (services_access['good'] | food_access['good']), 
        liveability['very_high']
    ))
    
    # Super high quality rule - all factors are good
    rules.append(ctrl.Rule(
        transport_access['good'] & healthcare_access['good'] & 
        education_access['good'] & env_quality['good'] &
        (services_access['good'] | food_access['good'] | recreation_access['good']),
        liveability['very_high']
    ))
    
    rules.append(ctrl.Rule(
        transport_access['poor'] & 
        (healthcare_access['poor'] | education_access['poor']), 
        liveability['very_low']
    ))
    
    # Create and return control system
    liveability_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(liveability_ctrl)

def evaluate_liveability_geocoded(user_lat, user_lon, location_data, fuzzy_system=None):
    """
<<<<<<< HEAD
    Evaluate liveability score using fuzzy logic for a specific geocoded location.
    This version assumes the area_name is already determined.
    """
    print(f"\nEvaluating liveability for coordinates: ({user_lat:.6f}, {user_lon:.6f})")
    
    # --- THIS SECTION IS REMOVED, as analyze_location_liveability handles it ---
    # area_name = determine_area_from_coordinates(user_lat, user_lon)
    # print(f"Location determined as: {area_name}")
=======
    Evaluate liveability score using fuzzy logic for a specific geocoded location
    
    Args:
        user_lat: User's latitude coordinate
        user_lon: User's longitude coordinate
        location_data: Dictionary with dataframes for each category
        fuzzy_system: Preconfigured fuzzy system (if None, creates a new one)
    
    Returns:
        Dictionary with overall score and category-specific scores
    """
    print(f"\nEvaluating liveability for coordinates: ({user_lat:.6f}, {user_lon:.6f})")
    
    # Determine which area these coordinates are in
    area_name = determine_area_from_coordinates(user_lat, user_lon)
    print(f"Location determined as: {area_name}")
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a
    
    # Calculate accessibility scores for each category using geocoded distances
    category_scores = {}
    
    for category, df in location_data.items():
        score = calculate_category_score_geocoded(df, category, user_lat, user_lon)
        category_scores[category] = score
    
    # Ensure we have scores for all required categories
    required_categories = ['Transport', 'Health', 'Education', 'Services', 'Recreation', 'Food', 'Environment']
    for category in required_categories:
        if category not in category_scores:
            print(f"Warning: Missing data for {category}, setting score to 0")
            category_scores[category] = 0.0
    
    # Create or use fuzzy system
    if fuzzy_system is None:
        fuzzy_system = define_fuzzy_system()
    
    # Input category scores into fuzzy system
    fuzzy_system.input['transport_access'] = category_scores.get('Transport', 0)
    fuzzy_system.input['healthcare_access'] = category_scores.get('Health', 0)
    fuzzy_system.input['education_access'] = category_scores.get('Education', 0)
    fuzzy_system.input['services_access'] = category_scores.get('Services', 0)
    fuzzy_system.input['recreation_access'] = category_scores.get('Recreation', 0)
    fuzzy_system.input['food_access'] = category_scores.get('Food', 0)
    fuzzy_system.input['env_quality'] = category_scores.get('Environment', 0)
    
    # Compute fuzzy liveability score
    try:
        fuzzy_system.compute()
        liveability_score = fuzzy_system.output['liveability']
    except Exception as e:
        print(f"Error in fuzzy computation: {e}")
        # Fallback to weighted average if fuzzy system fails
        weights = {
<<<<<<< HEAD
            'Services': 0.25, 'Health': 0.10, 'Education': 0.10, 'Recreation': 0.03,
            'Food': 0.02, 'Economic_Services': 0.05, 'Transport': 0.20, 'Environment': 0.25
=======
            'Transport': 0.20, 
            'Health': 0.15, 
            'Education': 0.15,
            'Services': 0.10, 
            'Recreation': 0.10, 
            'Food': 0.10,
            'Environment': 0.20
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a
        }
        liveability_score = sum(category_scores.get(cat, 0) * weights.get(cat, 0) for cat in weights) * 10
    
    # Return results
    return {
        'overall_score': liveability_score,
        'category_scores': category_scores,
<<<<<<< HEAD
        'coordinates': (user_lat, user_lon)
        # The 'area': area_name line is removed
=======
        'coordinates': (user_lat, user_lon),
        'area': area_name
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a
    }

def evaluate_liveability(location_data, fuzzy_system=None):
    """
    Evaluate liveability score using fuzzy logic based on accessibility scores
    for various categories of urban amenities.
    
    Args:
        location_data: Dictionary with dataframes for each category
        fuzzy_system: Preconfigured fuzzy system (if None, creates a new one)
    
    Returns:
        Dictionary with overall score and category-specific scores
    """
    # Calculate accessibility scores for each category
    category_scores = {}
    
    for category, df in location_data.items():
        score = calculate_category_score(df, category)
        category_scores[category] = score
        print(f"{category} accessibility score: {score:.2f}/10")
    
    # Ensure we have scores for all required categories
    required_categories = ['Transport', 'Health', 'Education', 'Services', 'Recreation', 'Food', 'Environment']
    for category in required_categories:
        if category not in category_scores:
            print(f"Warning: Missing data for {category}, setting score to 0")
            category_scores[category] = 0.0
    
    # Create or use fuzzy system
    if fuzzy_system is None:
        fuzzy_system = define_fuzzy_system()
    
    # Input category scores into fuzzy system
    fuzzy_system.input['transport_access'] = category_scores.get('Transport', 0)
    fuzzy_system.input['healthcare_access'] = category_scores.get('Health', 0)
    fuzzy_system.input['education_access'] = category_scores.get('Education', 0)
    fuzzy_system.input['services_access'] = category_scores.get('Services', 0)
    fuzzy_system.input['recreation_access'] = category_scores.get('Recreation', 0)
    fuzzy_system.input['food_access'] = category_scores.get('Food', 0)
    fuzzy_system.input['env_quality'] = category_scores.get('Environment', 0)  # Added environmental quality
    
    # Compute fuzzy liveability score
    try:
        fuzzy_system.compute()
        liveability_score = fuzzy_system.output['liveability']
    except Exception as e:
        print(f"Error in fuzzy computation: {e}")
        # Fallback to weighted average if fuzzy system fails
        weights = {
<<<<<<< HEAD
    # Institutional (0.25 total)
    'Services': 0.25,

    # Social (0.25 total)
    'Health': 0.10,
    'Education': 0.10,
    'Recreation': 0.03,
    'Food': 0.02,

    # Economic (0.05 total)
    'Economic_Services': 0.05,  # Only used if category exists

    # Physical (0.45 total)
    'Transport': 0.20,
    'Environment': 0.25
}

=======
            'Transport': 0.20, 
            'Health': 0.15, 
            'Education': 0.15,
            'Services': 0.10, 
            'Recreation': 0.10, 
            'Food': 0.10,
            'Environment': 0.20  # Environmental quality has high weight in fallback
        }
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a
        liveability_score = sum(category_scores.get(cat, 0) * weights.get(cat, 0) for cat in weights) * 10
    
    # Return results
    return {
        'overall_score': liveability_score,
        'category_scores': category_scores
    }

# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

def visualize_liveability_scores(results, location_name='Location'):
    """
    Create visualization of liveability results
    
    Args:
        results: Dictionary with overall_score and category_scores
        location_name: Name of the location for plot title
    """
    # Extract scores
    overall_score = results['overall_score']
    category_scores = results['category_scores']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create bar chart for category scores
    categories = list(category_scores.keys())
    scores = list(category_scores.values())
    
    # Set colormap based on scores
    colors = plt.cm.RdYlGn(np.array(scores) / 10)
    
    # Plot bar chart
    ax1.bar(categories, scores, color=colors)
    ax1.set_ylim(0, 10)
    ax1.set_ylabel('Accessibility Score (0-10)')
    ax1.set_title('Category Accessibility Scores')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add score values on top of bars
    for i, score in enumerate(scores):
        ax1.text(i, score + 0.2, f"{score:.1f}", ha='center')
    
    # Create radar chart for category scores
    categories = list(category_scores.keys())
    scores = list(category_scores.values())
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
    # Add the first element at the end to close the radar chart
    categories_radar = categories + [categories[0]]
    scores_radar = scores + [scores[0]]
    angles_radar = angles + [angles[0]]  # Close the loop
    
    # Plot radar chart
    ax2.plot(angles_radar, scores_radar, 'o-', linewidth=2)
    ax2.fill(angles_radar, scores_radar, alpha=0.25)
    ax2.set_ylim(0, 10)
    ax2.set_xticks(angles)
    ax2.set_xticklabels(categories)
    ax2.set_title('Category Accessibility Radar Chart')
    
    # Add overall score as text
    plt.figtext(0.5, 0.01, f'Overall Liveability Score: {overall_score:.1f}/100', 
                ha='center', fontsize=14, 
                bbox=dict(facecolor='green' if overall_score > 70 else 'orange' if overall_score > 40 else 'red', 
                          alpha=0.2))
    
    plt.suptitle(f'Liveability Analysis for {location_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', f'liveability_{location_name.lower().replace(" ", "_")}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    
    # Create environmental quality breakdown visualization if available
    if 'Environment' in category_scores and category_scores['Environment'] > 0:
        plt.figure(figsize=(10, 6))
        plt.title(f'Environmental Quality Breakdown for {location_name}')
        
        # Create a pie chart showing relative importance of environmental factors
        env_factors = {
            'Air Quality': 0.5,
            'Noise Pollution': 0.2,
            'Green Spaces': 0.2,
            'Water Quality': 0.1
        }
        
        plt.pie(env_factors.values(), labels=env_factors.keys(), autopct='%1.1f%%', 
                startangle=90, colors=plt.cm.Greens([0.3, 0.5, 0.7, 0.9]))
        plt.axis('equal')
        
        env_score = category_scores['Environment']
        plt.figtext(0.5, 0.01, f'Environmental Quality Score: {env_score:.1f}/10', 
                    ha='center', fontsize=14, 
                    bbox=dict(facecolor='green' if env_score > 7 else 'orange' if env_score > 4 else 'red', 
                              alpha=0.2))
        
        # Save environmental figure
        env_output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 
                                       f'environment_{location_name.lower().replace(" ", "_")}.png')
        plt.savefig(env_output_path)
        print(f"Environmental visualization saved to {env_output_path}")
    
    # Display in notebook if running in interactive mode
    plt.show()

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def analyze_geocode_liveability(latitude, longitude, data_path=None):
    """
    Main function to analyze liveability for a specific geocoded location
    
    Args:
        latitude: Exact latitude coordinate
        longitude: Exact longitude coordinate  
        data_path: Path to data files (if None, uses default path based on determined area)
    
    Returns:
        Dictionary with liveability scores and saves visualization
    """
    print(f"\n{'='*60}")
    print(f"GEOCODED LIVEABILITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Coordinates: {latitude:.6f}, {longitude:.6f}")
    
    # Determine which area/dataset to use based on coordinates
    area_name = determine_area_from_coordinates(latitude, longitude)
    
    # Set data path if not provided
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', area_name.lower())
    
    print(f"Using data from: {area_name}")
    print(f"Data path: {data_path}")
    
    # Load data for all categories
    location_data = load_location_data(area_name, data_path)
    
    if not location_data:
        print("Error: No data loaded. Please check data files exist.")
        return None
    
    # Define fuzzy system
    fuzzy_system = define_fuzzy_system()
    
    # Evaluate liveability for the specific coordinates
    results = evaluate_liveability_geocoded(latitude, longitude, location_data, fuzzy_system)
    
    # Print detailed results
    print(f"\n{'='*60}")
    print(f"RESULTS FOR COORDINATES ({latitude:.6f}, {longitude:.6f})")
    print(f"{'='*60}")
    print(f"Overall Liveability Score: {results['overall_score']:.1f}/100")
    print(f"\nCategory Accessibility Scores (0-10 scale):")
    for category, score in results['category_scores'].items():
        print(f"  {category:12}: {score:.1f}/10")
    
    # Create location name for visualization
    location_name = f"{area_name}_{latitude:.4f}_{longitude:.4f}"
    
    # Visualize results
    visualize_liveability_scores(results, location_name)
    
    return results

<<<<<<< HEAD
def get_current_location_demo(data_path=DATA_DIR):
    """
    Automatically extract sample coordinates from available CSV files.
    
    This replaces the hardcoded Jhalwa samples.
    It scans all datasets, extracts real latitude/longitude values,
    and returns a representative (mean/central) coordinate.
    """
    import glob
    
    all_coords = []

    # Scan all csv files inside the directory
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))

    if not csv_files:
        print("No CSV files found in data directory!")
        return None

    # Extract coordinates from each CSV
    for file in csv_files:
        try:
            df = pd.read_csv(file)

            if "Latitude" in df.columns and "Longitude" in df.columns:
                # Keep only valid coordinates
                for lat, lon in zip(df["Latitude"], df["Longitude"]):
                    if pd.notna(lat) and pd.notna(lon):
                        all_coords.append((lat, lon))
        except Exception as e:
            print(f"Could not read file {file}: {e}")

    if not all_coords:
        print("No valid coordinates found in dataset.")
        return None

    # Compute central coordinate using mean (centroid)
    latitudes = [c[0] for c in all_coords]
    longitudes = [c[1] for c in all_coords]

    center_lat = float(np.mean(latitudes))
    center_lon = float(np.mean(longitudes))

    # Print available locations (sample)
    print("Extracted location samples:")
    for i, (lat, lon) in enumerate(all_coords[:10], 1):  # Show only first 10
        print(f"  {i}. Coordinates: ({lat:.4f}, {lon:.4f})")

    print(f"\nAuto-selected central coordinate: ({center_lat:.4f}, {center_lon:.4f})")

    return center_lat, center_lon


def analyze_location_liveability(location_name, data_path=None):
    """
    Main function to analyze liveability for a given location.
    This version FINDS THE CENTER of the area and calculates distances.
=======
def get_current_location_demo():
    """
    Demo function that simulates getting current location
    In a real application, this would use GPS or user input
    
    Returns sample coordinates within Jhalwa area
    """
    # Sample coordinates within Jhalwa, Prayagraj
    # In reality, you'd get these from GPS or user input
    demo_coordinates = [
        (25.4358, 81.8463),  # Sample location 1
        (25.4420, 81.8520),  # Sample location 2  
        (25.4380, 81.8480),  # Sample location 3
        (25.4400, 81.8500),  # Sample location 4 (central)
    ]
    
    print("Demo locations available:")
    for i, (lat, lon) in enumerate(demo_coordinates, 1):
        print(f"  {i}. Coordinates: ({lat:.4f}, {lon:.4f})")
    
    # Return the central location as default
    return demo_coordinates[3]  # Central location

def analyze_location_liveability(location_name, data_path=None):
    """
    Main function to analyze liveability for a given location
    
    Args:
        location_name: Name of the location
        data_path: Path to data files (if None, uses default path)
    
    Returns:
        Dictionary with liveability scores and saves visualization
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a
    """
    # Set data path if not provided
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', location_name.lower())
    
    # Load data for all categories
    location_data = load_location_data(location_name, data_path)
    
<<<<<<< HEAD
    # --- START NEW LOGIC ---
    # We need to find the "center" of this location to calculate distances
    all_coords = []
    for df in location_data.values():
        if "Latitude" in df.columns and "Longitude" in df.columns:
            for lat, lon in zip(df["Latitude"], df["Longitude"]):
                if pd.notna(lat) and pd.notna(lon):
                    all_coords.append((lat, lon))

    if not all_coords:
        print("❌ No valid coordinates found in any file for this location. Cannot calculate distances.")
        # Return empty/zero results so the main loop can continue
        return {
            'overall_score': 0,
            'category_scores': {
                "Transport": 0, "Health": 0, "Education": 0, "Services": 0,
                "Recreation": 0, "Food": 0, "Environment": 0, "Religion": 0
            }
        }

    # Compute central coordinate using mean (centroid)
    center_lat = float(np.mean([c[0] for c in all_coords]))
    center_lon = float(np.mean([c[1] for c in all_coords]))

    print(f"Calculated center for {location_name}: ({center_lat:.4f}, {center_lon:.4f})")
    
    # Define fuzzy system
    fuzzy_system = define_fuzzy_system()
    
    # Evaluate liveability using the geocoded function
    # This will calculate distances from the center point
    results = evaluate_liveability_geocoded(center_lat, center_lon, location_data, fuzzy_system)
    # --- END NEW LOGIC ---
=======
    # Define fuzzy system
    fuzzy_system = define_fuzzy_system()
    
    # Evaluate liveability
    results = evaluate_liveability(location_data, fuzzy_system)
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a
    
    # Print overall score
    print(f"\nOverall Liveability Score for {location_name}: {results['overall_score']:.1f}/100")
    
    # Visualize results
<<<<<<< HEAD
    # visualize_liveability_scores(results, location_name)
=======
    visualize_liveability_scores(results, location_name)
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a
    
    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================
<<<<<<< HEAD
def main():
    """
    Generalized main function to analyze ALL areas in the data folder
    and save the results to a single CSV file.
    """

    print("\n==============================================")
    print("   BATCH URBAN LIVEABILITY ASSESSMENT SYSTEM")
    print("==============================================\n")

    # --- 1. Define Paths ---
    # This path logic comes from your original script
    base_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    output_file = 'fuzzy_scores.csv' # The output file we need

    # --- 2. Detect all area folders in /data/ ---
    try:
        area_folders = [
            name for name in os.listdir(base_data_path)
            if os.path.isdir(os.path.join(base_data_path, name))
        ]
    except FileNotFoundError:
        print(f"❌ Error: Base data path not found: {base_data_path}")
        print("Please check the DATA_DIR path at the top of your script.")
        return

    if not area_folders:
        print(f"❌ No area folders found inside {base_data_path}")
        return
    
    print(f"📍 Found {len(area_folders)} areas to analyze...")
    print(area_folders)

    all_results_list = [] # This will store the data for our CSV

    # --- 3. Loop through and run analysis for ALL areas ---
    for selected_area in area_folders:
        
        print(f"\n🔎 Analyzing area: {selected_area}...")
        
        area_path = os.path.join(base_data_path, selected_area)
        
        # This is your original function that runs the fuzzy logic
        # It will print its own detailed logs as it runs
        results = analyze_location_liveability(selected_area, data_path=area_path)

        if not results:
            print(f"❌ Analysis failed for {selected_area}. Skipping.")
            continue

        # --- 4. Prepare data row for CSV ---
        # We create a flat dictionary for the CSV row
        row_data = {
            'place_name': selected_area,
            'Fuzzy_Score': round(results['overall_score'], 2)
        }
        
        # Add all individual category scores (e.g., 'Health_Score', 'Transport_Score')
        for category, score in results['category_scores'].items():
            row_data[f"{category}_Score"] = round(score, 2)
            
        all_results_list.append(row_data)
        
        print(f"--- Completed {selected_area}. Overall Score: {results['overall_score']:.1f}/100 ---")

    # --- 5. Save all results to a single CSV file using pandas ---
    if not all_results_list:
        print("❌ No results were generated.")
        return
        
    print(f"\n✅ All analyses complete. Saving {len(all_results_list)} results to {output_file}...")
    
    # Convert our list of dictionaries into a pandas DataFrame
    results_df = pd.DataFrame(all_results_list)
    
    # Reorder columns to be nice: place_name, Fuzzy_Score, then the rest
    cols_to_order = ['place_name', 'Fuzzy_Score']
    other_cols = sorted([col for col in results_df.columns if col not in cols_to_order])
    results_df = results_df[cols_to_order + other_cols]
    
    # Fill any missing category columns with 0 (in case one area was missing a file)
    results_df = results_df.fillna(0)
    
    try:
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"🎉 Successfully saved all fuzzy scores to {output_file}")
    
    except Exception as e:
        print(f"🚨 Error writing to CSV: {e}")

    # No return needed, we just saved the file
    return


if __name__ == "__main__":
    main()
=======

if __name__ == "__main__":
    # Set location to analyze
    location = "Jhalwa"
    
    # Run analysis
    results = analyze_location_liveability(location)
    
    # Print detailed results
    print("\nDetailed Results:")
    print(f"Overall Liveability Score: {results['overall_score']:.1f}/100")
    print("\nCategory Scores (0-10 scale):")
    for category, score in results['category_scores'].items():
        print(f"{category}: {score:.1f}")

# This function was causing conflicts and has been removed
# The proper evaluate_liveability function is defined earlier in the file

# =============================================================================
# TESTING UTILITIES
# =============================================================================

def run_sensitivity_analysis(base_location_data, parameter_ranges=None, n_samples=10):
    """
    Run sensitivity analysis by varying input parameters and measuring impact on output.
    
    Args:
        base_location_data: Base case location data
        parameter_ranges: Dictionary with ranges for parameters to vary
        n_samples: Number of samples to take within each range
    
    Returns:
        DataFrame with sensitivity analysis results
    """
    if parameter_ranges is None:
        parameter_ranges = {
            'transport_distance': (0.5, 5.0),
            'healthcare_distance': (0.5, 5.0),
            'education_distance': (0.5, 5.0),
            'services_distance': (0.5, 5.0),
            'recreation_distance': (0.5, 5.0),
            'food_distance': (0.5, 5.0),
        }
    
    # Initialize results
    results = []
    
    # Create fuzzy system
    fuzzy_system = define_fuzzy_system()
    
    # Run sensitivity analysis
    print("Running sensitivity analysis...")
    for param, (min_val, max_val) in parameter_ranges.items():
        for i in range(n_samples):
            # Create a copy of the base data
            modified_data = {k: v.copy() for k, v in base_location_data.items()}
            
            # Modify the parameter
            param_val = min_val + (max_val - min_val) * i / (n_samples - 1)
            category = param.split('_')[0].capitalize()
            
            if category in modified_data:
                # Apply modification to distances
                modified_data[category]['Distance_km'] = modified_data[category]['Distance_km'] * param_val
            
            # Evaluate liveability with modified data
            eval_result = evaluate_liveability(modified_data, fuzzy_system)
            
            # Store results
            results.append({
                'parameter': param,
                'value': param_val,
                'overall_score': eval_result['overall_score'],
                **{f"{k}_score": v for k, v in eval_result['category_scores'].items()}
            })
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def visualize_sensitivity(sensitivity_df):
    """Visualize sensitivity analysis results"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Plot sensitivity of overall score to each parameter
    plt.figure(figsize=(12, 8))
    for param in sensitivity_df['parameter'].unique():
        param_df = sensitivity_df[sensitivity_df['parameter'] == param]
        plt.plot(param_df['value'], param_df['overall_score'], 'o-', label=param)
    
    plt.xlabel('Parameter Value (Scaling Factor)')
    plt.ylabel('Overall Liveability Score')
    plt.title('Sensitivity of Liveability Score to Parameter Changes')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'sensitivity_analysis.png')
    plt.savefig(output_path)
    print(f"Sensitivity analysis saved to {output_path}")
    
    # Display in notebook if running in interactive mode
    plt.show()

# Remove references to old visualize_results function - we're using visualize_liveability_scores now

# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def validate_model(test_cases=None):
    """
    Validate the model against known test cases
    
    Args:
        test_cases: List of test cases with known inputs and expected outputs
        
    Returns:
        DataFrame with validation results
    """
    if test_cases is None:
        # Define some standard test cases
        test_cases = [
            {
                'name': 'High Quality Urban Area',
                'inputs': {
                    'Transport': pd.DataFrame({'Distance_km': [0.5, 0.8, 1.2]}),
                    'Health': pd.DataFrame({'Distance_km': [0.7, 1.5]}),
                    'Education': pd.DataFrame({'Distance_km': [0.3, 1.0, 1.5]}),
                    'Services': pd.DataFrame({'Distance_km': [0.2, 0.6, 1.0]}),
                    'Recreation': pd.DataFrame({'Distance_km': [0.5, 1.2]}),
                    'Food': pd.DataFrame({'Distance_km': [0.3, 0.8, 1.2]})
                },
                'expected_score_range': (75, 100)
            },
            {
                'name': 'Medium Quality Area',
                'inputs': {
                    'Transport': pd.DataFrame({'Distance_km': [1.5, 2.0, 3.0]}),
                    'Health': pd.DataFrame({'Distance_km': [2.0, 4.0]}),
                    'Education': pd.DataFrame({'Distance_km': [1.5, 2.5, 3.5]}),
                    'Services': pd.DataFrame({'Distance_km': [1.2, 2.0, 3.0]}),
                    'Recreation': pd.DataFrame({'Distance_km': [1.5, 3.0]}),
                    'Food': pd.DataFrame({'Distance_km': [1.0, 2.0, 3.0]})
                },
                'expected_score_range': (40, 75)
            },
            {
                'name': 'Low Quality Area',
                'inputs': {
                    'Transport': pd.DataFrame({'Distance_km': [4.0, 5.0, 7.0]}),
                    'Health': pd.DataFrame({'Distance_km': [5.0, 8.0]}),
                    'Education': pd.DataFrame({'Distance_km': [4.0, 6.0, 9.0]}),
                    'Services': pd.DataFrame({'Distance_km': [3.5, 5.0, 7.0]}),
                    'Recreation': pd.DataFrame({'Distance_km': [4.0, 7.0]}),
                    'Food': pd.DataFrame({'Distance_km': [3.0, 5.0, 8.0]})
                },
                'expected_score_range': (0, 40)
            }
        ]
    
    results = []
    
    for test_case in test_cases:
        print(f"Validating test case: {test_case['name']}")
        
        # Evaluate liveability for this test case
        test_result = evaluate_liveability(test_case['inputs'])
        
        # Check if score is within expected range
        score = test_result['overall_score']
        min_expected, max_expected = test_case['expected_score_range']
        is_valid = min_expected <= score <= max_expected
        
        results.append({
            'name': test_case['name'],
            'score': score,
            'min_expected': min_expected,
            'max_expected': max_expected,
            'is_valid': is_valid
        })
        
        print(f"  Score: {score:.1f}, Expected: {min_expected}-{max_expected}, Valid: {is_valid}")
    
    return pd.DataFrame(results)

def main():
    """Main function to run the liveability assessment"""
    print("Urban Liveability Assessment for Jhalwa, Prayagraj")
    print("=" * 60)
    
    # Option 1: Area-wide analysis (original functionality)
    print("\n1. AREA-WIDE ANALYSIS")
    print("-" * 30)
    results_area = analyze_location_liveability("Jhalwa")
    
    # Option 2: Geocoded analysis for specific coordinates  
    print("\n\n2. GEOCODED ANALYSIS")
    print("-" * 30)
    
    # Get current location (demo coordinates)
    current_lat, current_lon = get_current_location_demo()
    print(f"Analyzing current location: ({current_lat:.6f}, {current_lon:.6f})")
    
    # Run geocoded analysis
    results_geocoded = analyze_geocode_liveability(current_lat, current_lon)
    
    # Compare results
    if results_geocoded:
        print("\n\n3. COMPARISON")
        print("-" * 30)
        print(f"Area-wide score for Jhalwa: {results_area['overall_score']:.1f}/100")
        print(f"Geocoded score for ({current_lat:.4f}, {current_lon:.4f}): {results_geocoded['overall_score']:.1f}/100")
        
        diff = results_geocoded['overall_score'] - results_area['overall_score']
        if diff > 0:
            print(f"This specific location is {diff:.1f} points BETTER than area average")
        elif diff < 0:
            print(f"This specific location is {abs(diff):.1f} points WORSE than area average")
        else:
            print("This specific location matches the area average")

if __name__ == "__main__":
    main()
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a
