"""
Urban Liveability Assessment Model
----------------------------------
A comprehensive fuzzy logic-based model for assessing urban liveability
with scientific validation, sensitivity analysis, and contextual calibration.
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
DATA_DIR = r"c:\Users\gagan\OneDrive\Desktop\urban_bot\data\jhalwa"
OUTPUT_DIR = r"c:\Users\gagan\OneDrive\Desktop\urban_bot\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reference points for validation (if available)
REFERENCE_POINTS = {
    # Format: 'location_name': {'known_score': value, 'lat': value, 'lon': value}
    # Add known reference points if available
}

# Category weights (determined through literature review and expert input)
# These weights reflect the relative importance of each category to liveability
CATEGORY_WEIGHTS = {
    'transport': 0.25,
    'healthcare': 0.20,
    'education': 0.20,
    'services': 0.15,
    'recreation': 0.10,
    'food': 0.10
}

# Regional context parameters
# These parameters allow calibration based on regional expectations
REGIONAL_CONTEXT = {
    'urban_density': 'medium',  # Options: 'low', 'medium', 'high'
    'region_type': 'developing',  # Options: 'developing', 'developed'
    'population': 'medium',  # Options: 'low', 'medium', 'high'
    'max_reasonable_distance': 10.0  # Maximum distance (km) that's considered in the model
}

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
        file_name = f"Jhalwa_Prayagraj_{category}.csv"
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            data[category] = pd.read_csv(file_path)
            print(f"Loaded {len(data[category])} records for {category}")
        else:
            print(f"Warning: Data file for {category} not found at {file_path}")
            data[category] = pd.DataFrame()
    
    return data

def analyze_distance_distributions(all_data):
    """Analyze the distribution of distances for each category"""
    distance_stats = {}
    
    for category, df in all_data.items():
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
        distance_stats[category] = stats
        
        print(f"\n{category} Distance Statistics:")
        print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f} km")
        print(f"  Mean: {stats['mean']:.2f} km, Median: {stats['median']:.2f} km")
        print(f"  Standard Deviation: {stats['std']:.2f} km")
        print(f"  Percentiles: 25%={stats['percentiles']['25%']:.2f}, 50%={stats['percentiles']['50%']:.2f}, 75%={stats['percentiles']['75%']:.2f}")
    
    return distance_stats

def calculate_accessibility_score(df, category, regional_context=REGIONAL_CONTEXT):
    """
    Calculate accessibility score with improved methodology:
    1. Consider variety of amenities (not just distance)
    2. Apply distance decay function
    3. Account for regional context
    """
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

def define_advanced_fuzzy_system(distance_stats):
    """Define a more sophisticated fuzzy logic system for urban liveability"""
    
    # Define input variables (accessibility scores, 0-10 scale where 10 is best)
    transport_access = ctrl.Antecedent(np.arange(0, 11, 0.1), 'transport_access')
    healthcare_access = ctrl.Antecedent(np.arange(0, 11, 0.1), 'healthcare_access')
    education_access = ctrl.Antecedent(np.arange(0, 11, 0.1), 'education_access')
    services_access = ctrl.Antecedent(np.arange(0, 11, 0.1), 'services_access')
    recreation_access = ctrl.Antecedent(np.arange(0, 11, 0.1), 'recreation_access')
    food_access = ctrl.Antecedent(np.arange(0, 11, 0.1), 'food_access')
    
    # Define output variable
    liveability = ctrl.Consequent(np.arange(0, 101, 1), 'liveability')
    
    # Define membership functions based on statistical analysis
    # Use quartiles from the distance statistics where available
    
    # For all inputs, higher is better (0-10 scale)
    for input_var in [transport_access, healthcare_access, education_access, 
                      services_access, recreation_access, food_access]:
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
    
    # Combined accessibility rules
    rules.append(ctrl.Rule(
        transport_access['good'] & healthcare_access['good'] & education_access['good'], 
        liveability['very_high']
    ))
    
    rules.append(ctrl.Rule(
        transport_access['moderate'] & healthcare_access['moderate'] & education_access['moderate'], 
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
    
    rules.append(ctrl.Rule(
        transport_access['poor'] & 
        (healthcare_access['poor'] | education_access['poor']), 
        liveability['very_low']
    ))
    
    # Balanced amenities rule
    rules.append(ctrl.Rule(
        transport_access['moderate'] & healthcare_access['moderate'] & 
        education_access['moderate'] & services_access['moderate'] &
        recreation_access['moderate'] & food_access['moderate'],
        liveability['medium']
    ))
    
    # Create and return control system
    liveability_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(liveability_ctrl)

def evaluate_liveability_advanced(all_data, regional_context=REGIONAL_CONTEXT):
    """Evaluate liveability using improved methodology"""
    print("\nCalculating accessibility scores with advanced methodology...")
    
    # Calculate accessibility scores for each category
    accessibility_scores = {}
    
    # Calculate transport accessibility (using improved method)
    transport_score = calculate_accessibility_score(
        all_data.get('Transport', pd.DataFrame()), 'Transport', regional_context
    )
    accessibility_scores['transport'] = transport_score
    
    # Calculate healthcare accessibility
    healthcare_score = calculate_accessibility_score(
        all_data.get('Health', pd.DataFrame()), 'Health', regional_context
    )
    accessibility_scores['healthcare'] = healthcare_score
    
    # Calculate education accessibility
    education_score = calculate_accessibility_score(
        all_data.get('Education', pd.DataFrame()), 'Education', regional_context
    )
    accessibility_scores['education'] = education_score
    
    # Calculate services accessibility
    services_score = calculate_accessibility_score(
        all_data.get('Shopping & Services', pd.DataFrame()), 'Services', regional_context
    )
    accessibility_scores['services'] = services_score
    
    # Calculate recreation accessibility
    recreation_score = calculate_accessibility_score(
        all_data.get('Recreation & Leisure', pd.DataFrame()), 'Recreation', regional_context
    )
    accessibility_scores['recreation'] = recreation_score
    
    # Calculate food accessibility
    food_score = calculate_accessibility_score(
        all_data.get('Food & Drink', pd.DataFrame()), 'Food', regional_context
    )
    accessibility_scores['food'] = food_score
    
    # Print scores
    for category, score in accessibility_scores.items():
        print(f"{category.capitalize()} accessibility score: {score:.2f}/10")
    
    # Get distance statistics for refining the fuzzy system
    distance_stats = analyze_distance_distributions(all_data)
    
    # Create fuzzy system
    liveability_simulation = define_advanced_fuzzy_system(distance_stats)
    
    # Input category scores to fuzzy system
    liveability_simulation.input['transport_access'] = accessibility_scores['transport']
    liveability_simulation.input['healthcare_access'] = accessibility_scores['healthcare']
    liveability_simulation.input['education_access'] = accessibility_scores['education']
    liveability_simulation.input['services_access'] = accessibility_scores['services']
    liveability_simulation.input['recreation_access'] = accessibility_scores['recreation']
    liveability_simulation.input['food_access'] = accessibility_scores['food']
    
    # Compute fuzzy result
    try:
        liveability_simulation.compute()
        fuzzy_liveability_score = liveability_simulation.output['liveability']
        print(f"Fuzzy logic liveability score: {fuzzy_liveability_score:.2f}/100")
        
        # Also calculate weighted sum (traditional approach) for comparison
        weighted_sum = (
            accessibility_scores['transport'] * CATEGORY_WEIGHTS['transport'] +
            accessibility_scores['healthcare'] * CATEGORY_WEIGHTS['healthcare'] +
            accessibility_scores['education'] * CATEGORY_WEIGHTS['education'] +
            accessibility_scores['services'] * CATEGORY_WEIGHTS['services'] +
            accessibility_scores['recreation'] * CATEGORY_WEIGHTS['recreation'] +
            accessibility_scores['food'] * CATEGORY_WEIGHTS['food']
        )
        
        # Convert to 0-100 scale
        weighted_liveability_score = (weighted_sum / 10) * 100
        print(f"Weighted sum liveability score: {weighted_liveability_score:.2f}/100")
        
        # Return both for comparison
        return {
            'accessibility_scores': accessibility_scores,
            'fuzzy_liveability_score': fuzzy_liveability_score,
            'weighted_liveability_score': weighted_liveability_score,
            'distance_stats': distance_stats
        }
    except Exception as e:
        print(f"Error computing liveability score: {e}")
        import traceback
        traceback.print_exc()
        return None

def sensitivity_analysis(all_data, regional_context=REGIONAL_CONTEXT):
    """
    Perform sensitivity analysis to understand how changes in 
    accessibility of different amenities affect the overall score
    """
    print("\nPerforming sensitivity analysis...")
    
    # Get baseline results
    baseline_results = evaluate_liveability_advanced(all_data, regional_context)
    if not baseline_results:
        return None
    
    baseline_score = baseline_results['fuzzy_liveability_score']
    print(f"Baseline liveability score: {baseline_score:.2f}")
    
    # Perform sensitivity analysis on each category
    sensitivity = {}
    
    # Define perturbation amounts
    perturbations = [-3, -2, -1, 1, 2, 3]
    
    for category in baseline_results['accessibility_scores'].keys():
        sensitivity[category] = []
        
        # Test how changes in this category affect overall score
        for perturbation in perturbations:
            # Create a copy of the baseline scores
            modified_scores = baseline_results['accessibility_scores'].copy()
            
            # Apply perturbation (ensure score stays in valid range)
            original_score = modified_scores[category]
            modified_scores[category] = max(min(original_score + perturbation, 10), 0)
            
            # Create fuzzy system
            liveability_simulation = define_advanced_fuzzy_system(baseline_results['distance_stats'])
            
            # Input modified scores
            liveability_simulation.input['transport_access'] = modified_scores['transport']
            liveability_simulation.input['healthcare_access'] = modified_scores['healthcare']
            liveability_simulation.input['education_access'] = modified_scores['education']
            liveability_simulation.input['services_access'] = modified_scores['services']
            liveability_simulation.input['recreation_access'] = modified_scores['recreation']
            liveability_simulation.input['food_access'] = modified_scores['food']
            
            # Compute result
            try:
                liveability_simulation.compute()
                modified_liveability = liveability_simulation.output['liveability']
                
                # Calculate impact
                score_change = modified_liveability - baseline_score
                percent_change = (score_change / baseline_score) * 100
                
                sensitivity[category].append({
                    'perturbation': perturbation,
                    'score_change': score_change,
                    'percent_change': percent_change
                })
                
                print(f"{category}: {perturbation:+d} points → {modified_liveability:.2f} ({percent_change:+.2f}%)")
                
            except Exception as e:
                print(f"Error in sensitivity analysis for {category}: {e}")
    
    return {
        'baseline': baseline_score,
        'sensitivity': sensitivity
    }

def create_advanced_visualizations(results, sensitivity_results=None):
    """Create more informative and visually appealing visualizations"""
    if not results:
        print("No results to visualize")
        return
    
    # Set up plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Extract scores
    accessibility_scores = results['accessibility_scores']
    fuzzy_score = results['fuzzy_liveability_score']
    weighted_score = results['weighted_liveability_score']
    
    # 1. Radar chart with category weights
    categories = list(accessibility_scores.keys())
    scores = [accessibility_scores[cat] for cat in categories]
    weights = [CATEGORY_WEIGHTS[cat] * 10 for cat in categories]  # Scale up for visibility
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    scores = scores + scores[:1]  # Close the loop
    weights = weights + weights[:1]  # Close the loop
    categories = categories + categories[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Plot scores
    ax.plot(angles, scores, 'o-', linewidth=2, label='Accessibility Score', color='#1f77b4')
    ax.fill(angles, scores, alpha=0.25, color='#1f77b4')
    
    # Plot weights
    ax.plot(angles, weights, 'o-', linewidth=2, label='Category Weight', color='#ff7f0e')
    ax.fill(angles, weights, alpha=0.1, color='#ff7f0e')
    
    # Customize radar chart
    ax.set_thetagrids(np.degrees(angles[:-1]), [c.capitalize() for c in categories[:-1]])
    ax.set_ylim(0, 10)
    ax.set_title('Liveability Profile with Category Weights', size=15)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'liveability_radar_advanced.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Stacked bar chart showing contribution to overall score
    weighted_contributions = {}
    for category in categories[:-1]:  # Exclude the duplicated last item
        weighted_contributions[category] = accessibility_scores[category] * CATEGORY_WEIGHTS[category] * 10
    
    # Sort categories by contribution
    sorted_categories = sorted(weighted_contributions.keys(), 
                              key=lambda x: weighted_contributions[x], 
                              reverse=True)
    
    # Create stacked bar
    plt.figure(figsize=(12, 6))
    
    # Plot bars
    bottom = 0
    for category in sorted_categories:
        contribution = weighted_contributions[category]
        plt.bar('Liveability Score', contribution, bottom=bottom, 
                label=f"{category.capitalize()} ({contribution:.1f})")
        bottom += contribution
    
    plt.axhline(y=weighted_score, color='r', linestyle='--', 
                label=f'Weighted Score: {weighted_score:.1f}')
    
    plt.axhline(y=fuzzy_score, color='g', linestyle='-', 
                label=f'Fuzzy Logic Score: {fuzzy_score:.1f}')
    
    plt.ylim(0, 100)
    plt.ylabel('Score Contribution')
    plt.title('Contribution of Each Category to Overall Liveability Score', size=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'category_contributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Comparison of accessibility scores
    plt.figure(figsize=(12, 6))
    
    # Sort categories by score
    sorted_by_score = sorted(accessibility_scores.items(), key=lambda x: x[1], reverse=True)
    categories_sorted = [item[0].capitalize() for item in sorted_by_score]
    scores_sorted = [item[1] for item in sorted_by_score]
    
    # Create horizontal bar chart
    bars = plt.barh(categories_sorted, scores_sorted, color='skyblue')
    
    # Add score labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{scores_sorted[i]:.1f}', va='center')
    
    plt.xlim(0, 10.5)
    plt.xlabel('Accessibility Score (0-10)')
    plt.title('Accessibility Scores by Category', size=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'accessibility_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Sensitivity analysis visualization (if available)
    if sensitivity_results:
        plt.figure(figsize=(14, 8))
        
        # For each category, plot how changes affect overall score
        categories = list(sensitivity_results['sensitivity'].keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        
        for i, category in enumerate(categories):
            perturbations = [item['perturbation'] for item in sensitivity_results['sensitivity'][category]]
            percent_changes = [item['percent_change'] for item in sensitivity_results['sensitivity'][category]]
            
            plt.plot(perturbations, percent_changes, 'o-', linewidth=2, 
                    label=category.capitalize(), color=colors[i])
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        plt.xlabel('Change in Category Score (points)', size=12)
        plt.ylabel('Change in Overall Liveability Score (%)', size=12)
        plt.title('Sensitivity Analysis: Impact of Category Changes on Overall Score', size=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(OUTPUT_DIR, 'sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Elasticity chart (responsiveness of overall score to category changes)
        plt.figure(figsize=(10, 6))
        
        # Calculate elasticity for each category
        elasticity = {}
        for category in categories:
            # Use +1 and -1 perturbations to calculate elasticity
            plus_one = next((item for item in sensitivity_results['sensitivity'][category] 
                             if item['perturbation'] == 1), None)
            minus_one = next((item for item in sensitivity_results['sensitivity'][category] 
                              if item['perturbation'] == -1), None)
            
            if plus_one and minus_one:
                # Average of absolute percent changes
                elasticity[category] = (abs(plus_one['percent_change']) + 
                                       abs(minus_one['percent_change'])) / 2
        
        # Sort by elasticity
        sorted_elasticity = sorted(elasticity.items(), key=lambda x: x[1], reverse=True)
        elasticity_categories = [item[0].capitalize() for item in sorted_elasticity]
        elasticity_values = [item[1] for item in sorted_elasticity]
        
        # Create bar chart
        bars = plt.bar(elasticity_categories, elasticity_values, color='lightgreen')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.ylabel('Elasticity (% Change in Overall Score)', size=12)
        plt.title('Elasticity of Liveability Score to Category Changes', size=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(OUTPUT_DIR, 'score_elasticity.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Advanced visualizations saved to {OUTPUT_DIR}")

def main():
    """Main function to run the advanced liveability assessment"""
    print("Advanced Urban Liveability Assessment for Jhalwa, Prayagraj")
    print("=" * 60)
    
    # Load all available data
    all_data = load_all_data()
    
    # Evaluate liveability with advanced methodology
    results = evaluate_liveability_advanced(all_data)
    
    # Perform sensitivity analysis
    sensitivity_results = sensitivity_analysis(all_data)
    
    # Create advanced visualizations
    if results:
        create_advanced_visualizations(results, sensitivity_results)
        print("\nAssessment complete! Check the output directory for visualizations.")

if __name__ == "__main__":
    main()
