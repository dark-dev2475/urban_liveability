"""
Demo script to demonstrate the Urban Liveability Model
"""
import os
import sys

# Add the src directory to path so we can import the module
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Try to import required packages, install if missing
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import skfuzzy as fuzz
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "pandas", "numpy", "matplotlib", "scikit-fuzzy", "seaborn"])
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import skfuzzy as fuzz

# Import our liveability module
from src.fuzzy_liveability import analyze_location_liveability

def create_sample_data():
    """Create sample data files if they don't exist"""
    data_dir = os.path.join(os.path.dirname(__file__), "data", "jhalwa")
    os.makedirs(data_dir, exist_ok=True)
    
    # Sample transport data
    transport_data = {
        'Category': ['Transport'] * 5,
        'Amenity': ['bus_stop', 'train_station', 'bus_stop', 'taxi', 'parking'],
        'Name': ['Bus Stop A', 'Railway Station', 'Bus Stop B', 'Taxi Stand', 'Parking Lot'],
        'Latitude': [25.4276, 25.4312, 25.4298, 25.4265, 25.4287],
        'Longitude': [81.7731, 81.7702, 81.7743, 81.7721, 81.7715],
        'Distance_km': [0.8, 3.2, 1.5, 1.1, 0.5],
        'Travel_Time_min': [10, 25, 15, 12, 5]
    }
    
    # Sample healthcare data
    health_data = {
        'Category': ['Health'] * 4,
        'Amenity': ['hospital', 'clinic', 'pharmacy', 'doctors'],
        'Name': ['General Hospital', 'City Clinic', 'Med Pharmacy', 'Dr. Sharma'],
        'Latitude': [25.4301, 25.4289, 25.4267, 25.4278],
        'Longitude': [81.7722, 81.7738, 81.7725, 81.7732],
        'Distance_km': [2.5, 1.2, 0.6, 1.0],
        'Travel_Time_min': [20, 12, 8, 10]
    }
    
    # Sample education data
    education_data = {
        'Category': ['Education'] * 5,
        'Amenity': ['university', 'college', 'school', 'school', 'library'],
        'Name': ['MNNIT', 'City College', 'Public School', 'Primary School', 'Central Library'],
        'Latitude': [25.4292, 25.4276, 25.4259, 25.4312, 25.4283],
        'Longitude': [81.7711, 81.7726, 81.7718, 81.7735, 81.7741],
        'Distance_km': [0.1, 1.8, 1.2, 1.5, 2.0],
        'Travel_Time_min': [2, 18, 15, 16, 20]
    }
    
    # Sample services data
    services_data = {
        'Category': ['Services'] * 6,
        'Amenity': ['supermarket', 'bank', 'post_office', 'atm', 'market', 'mall'],
        'Name': ['Big Mart', 'SBI Bank', 'Post Office', 'ATM', 'Local Market', 'City Mall'],
        'Latitude': [25.4281, 25.4269, 25.4295, 25.4271, 25.4283, 25.4310],
        'Longitude': [81.7729, 81.7721, 81.7735, 81.7723, 81.7718, 81.7705],
        'Distance_km': [0.7, 1.2, 1.6, 0.9, 0.5, 3.5],
        'Travel_Time_min': [8, 15, 18, 10, 8, 25]
    }
    
    # Sample recreation data
    recreation_data = {
        'Category': ['Recreation'] * 4,
        'Amenity': ['park', 'gym', 'cinema', 'playground'],
        'Name': ['City Park', 'Fitness Center', 'Movie Hall', 'Play Area'],
        'Latitude': [25.4285, 25.4273, 25.4307, 25.4268],
        'Longitude': [81.7725, 81.7719, 81.7712, 81.7731],
        'Distance_km': [1.0, 1.5, 2.8, 0.6],
        'Travel_Time_min': [12, 18, 22, 8]
    }
    
    # Sample food data
    food_data = {
        'Category': ['Food'] * 5,
        'Amenity': ['restaurant', 'cafe', 'fast_food', 'pub', 'tea_stall'],
        'Name': ['Family Restaurant', 'Coffee Shop', 'Burger Joint', 'City Pub', 'Tea Corner'],
        'Latitude': [25.4279, 25.4287, 25.4271, 25.4301, 25.4265],
        'Longitude': [81.7727, 81.7733, 81.7721, 81.7715, 81.7729],
        'Distance_km': [0.9, 1.2, 0.7, 2.3, 0.4],
        'Travel_Time_min': [10, 15, 8, 20, 5]
    }
    
    # Sample environmental data
    environment_data = {
        'Category': ['Environment'] * 15,
        'Pollutant': ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3'] * 3,
        'Name': ['Location 1'] * 5 + ['Location 2'] * 5 + ['Location 3'] * 5,
        'Latitude': [25.4276] * 5 + [25.4298] * 5 + [25.4312] * 5,
        'Longitude': [81.7731] * 5 + [81.7743] * 5 + [81.7702] * 5,
        'Value': [42, 86, 35, 12, 52, 36, 78, 42, 15, 60, 49, 92, 38, 18, 45],
        'Unit': ['µg/m³'] * 15,
        'Rating': ['Moderate', 'Poor', 'Good', 'Good', 'Moderate',
                   'Moderate', 'Moderate', 'Moderate', 'Good', 'Moderate',
                   'Poor', 'Poor', 'Moderate', 'Good', 'Moderate']
    }
    
    # Create and save dataframes
    data_files = {
        'Jhalwa_Prayagraj_Transport.csv': pd.DataFrame(transport_data),
        'Jhalwa_Prayagraj_Health.csv': pd.DataFrame(health_data),
        'Jhalwa_Prayagraj_Education.csv': pd.DataFrame(education_data),
        'Jhalwa_Prayagraj_Shopping & Services.csv': pd.DataFrame(services_data),
        'Jhalwa_Prayagraj_Recreation & Leisure.csv': pd.DataFrame(recreation_data),
        'Jhalwa_Prayagraj_Food & Drink.csv': pd.DataFrame(food_data),
        'Jhalwa_Prayagraj_Environment.csv': pd.DataFrame(environment_data)
    }
    
    for filename, df in data_files.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
            print(f"Created sample data file: {filename}")

def main():
    """Run the demo"""
    print("Urban Liveability Model Demo")
    print("============================")
    
    # Create sample data if needed
    create_sample_data()
    
    # Analyze liveability for Jhalwa
    print("\nAnalyzing liveability for Jhalwa, Prayagraj...")
    results = analyze_location_liveability("Jhalwa")
    
    print("\nDemonstration complete! Check the output directory for visualization.")

if __name__ == "__main__":
    main()
