<<<<<<< HEAD
# Urban Bot: Urban Liveability & Slum Detection System

A comprehensive machine learning system for assessing urban liveability and detecting slums using fuzzy logic, neural networks, and multi-dimensional urban features.

## 📋 Overview

This project combines **fuzzy logic-based accessibility analysis** with **deep learning classification** to evaluate urban areas across India. The system analyzes:

- **Distance-based accessibility** to amenities (transport, health, education, etc.)
- **Contextual urban features** (density, development level, infrastructure)
- **Sentiment analysis** from local reviews and news
- **Slum detection classification** using neural networks
- **Urban liveability scoring** using fuzzy inference systems

## 🎯 Core Features

1. **Fuzzy Logic Liveability Assessment**: Evaluates neighborhoods based on proximity to amenities with distance decay functions
2. **Neural Network Classification**: Predicts slum/non-slum areas using multi-feature training
3. **Multi-Source Data Integration**: Combines accessibility metrics, sentiment, and contextual features
4. **Comprehensive Reporting**: Generates visualizations and detailed liveability reports
5. **Scalable Analysis**: Supports 100+ Indian cities and municipalities

## 📁 Project Structure

```
urban_bot/
├── src/                          # Core modules
│   ├── fuzzy_liveability.py     # Fuzzy logic evaluation engine
│   ├── sentiment_features.py    # Sentiment analysis from reviews
│   ├── build_context_features.py # Urban context feature extraction
│   ├── data_builder.py          # Data pipeline
│   ├── decision_tree.py         # Decision tree analysis
│   ├── xgboost_model.py         # XGBoost classifier
│   └── compare.py               # Model comparison utilities
│
├── data/                         # Urban data by location
│   ├── jhalwa/
│   ├── Nagpur/
│   ├── Greater_Mumbai/
│   └── [100+ other locations]
│
├── models/                       # Trained model artifacts
│   └── [Saved model files]
│
├── output/                       # Generated reports & visualizations
│   └── [Liveability reports & charts]
│
├── Core Scripts
│   ├── predict_place.py         # 🚀 Main prediction interface
│   ├── neural_network.py        # Model training & optimization
│   ├── master_training_dataset.py # Feature aggregation pipeline
│   ├── demo_liveability.py      # Interactive demo with sample data
│   └── labels.csv               # Ground truth labels for training
│
├── Feature Files (Generated during pipeline)
│   ├── master_training_data.csv      # Combined training features
│   ├── fuzzy_scores.csv              # Fuzzy logic scores by location
│   ├── sentiment_features.csv        # Sentiment analysis results
│   ├── context_features.csv          # Urban context metrics
│   └── place_names_list.csv          # Reference location list
│
├── Model Artifacts
│   ├── liveability_model.keras   # Trained neural network
│   └── scaler.joblib             # Feature scaling parameters
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone and navigate to repository**:
   ```bash
   cd urban_bot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Dependencies include:
   - pandas, numpy, scikit-learn, tensorflow
   - scikit-fuzzy (for fuzzy logic)
   - matplotlib, seaborn (visualization)
   - newspaper3k, beautifulsoup4 (web scraping)
   - praw (Reddit API)
   - google-api-python-client (Google APIs)

## 🚀 Quick Start

### Option 1: Try the Interactive Demo
```bash
python demo_liveability.py
```
This creates sample data and demonstrates the liveability assessment system on a test location.

### Option 2: Predict Liveability for a Place
```bash
python predict_place.py
```
Follow the prompts to:
- Select a place from your dataset
- Get liveability scores
- View amenity accessibility analysis
- See comprehensive liveability report

## 📊 Workflow: How the System Works

### 1️⃣ Data Collection
- Gather amenity data (locations, distances) for target cities
- Collect local reviews, sentiment data, and contextual features

### 2️⃣ Feature Engineering
```bash
# Build context features (urban density, infrastructure, etc.)
python src/build_context_features.py

# Extract sentiment features from reviews
python src/sentiment_features.py

# Generate fuzzy logic scores (done in Stage 1)
# Results saved to: fuzzy_scores.csv
```

### 3️⃣ Feature Aggregation
```bash
# Combine all features into master training dataset
python master_training_dataset.py
# Output: master_training_data.csv
```

### 4️⃣ Model Training
```bash
# Train neural network classifier
python neural_network.py
# Outputs: liveability_model.keras, scaler.joblib
```

### 5️⃣ Prediction & Analysis
```bash
# Use trained model to predict and analyze new areas
python predict_place.py
```

## 📈 Key Modules

### `src/fuzzy_liveability.py`
Implements fuzzy logic inference for liveability assessment:
- Distance decay functions
- Amenity proximity scoring
- Fuzzy rule-based evaluation
- Multi-criteria aggregation

### `src/sentiment_features.py`
Extracts sentiment metrics from local sources:
- Reviews analysis
- News sentiment
- Social media signals
- Local perception scores

### `src/build_context_features.py`
Calculates urban context metrics:
- Population density
- Infrastructure level
- Development index
- Accessibility metrics

### `neural_network.py`
Deep learning model for slum detection:
- Feature scaling and normalization
- Class-weighted training (handles imbalanced data)
- Dropout regularization
- Confusion matrix and classification reports

## 📊 Output Files Generated

| File | Purpose |
|------|---------|
| `master_training_data.csv` | Combined features + labels for all locations |
| `fuzzy_scores.csv` | Liveability scores from fuzzy logic |
| `sentiment_features.csv` | Sentiment metrics per location |
| `context_features.csv` | Urban context features per location |
| `liveability_model.keras` | Trained neural network model |
| `scaler.joblib` | StandardScaler for feature normalization |
| `output/` | Generated liveability reports & visualizations |

## 🛠️ Configuration & Customization

### Adding New Locations
1. Create a new folder in `data/` with location name
2. Add amenity CSV files (Transport, Health, Education, etc.)
3. Run the feature pipeline
4. Retrain the model with new data

### Adjusting Model Parameters
Edit `neural_network.py`:
- Hidden layers, activation functions
- Dropout rates
- Learning rate and batch size
- Class weights for imbalanced data

### Tuning Fuzzy Logic Rules
Edit `src/fuzzy_liveability.py`:
- Distance decay parameters
- Amenity category weights
- Fuzzy membership functions
- Rule thresholds

## 📝 Data Format Requirements

### Amenity CSV Files (per location)
```
Category,Amenity,Name,Latitude,Longitude,Distance_km,Travel_Time_min
Transport,bus_stop,Bus Stop A,25.4276,81.7731,0.8,10
Health,hospital,General Hospital,25.4301,81.7702,3.2,25
```

### Labels File (`labels.csv`)
```
place_name,is_slum
jhalwa,1
nagpur_central,0
```

## 🤖 Model Performance

The system provides:
- **Classification metrics**: Precision, Recall, F1-Score
- **Confusion matrices** for error analysis
- **Liveability scores** on 0-100 scale
- **Amenity accessibility** breakdowns by category

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | Run `neural_network.py` first to train |
| "Missing feature files" | Run the feature pipeline: `master_training_dataset.py` |
| Import errors | Reinstall dependencies: `pip install -r requirements.txt --upgrade` |
| Out of memory | Reduce batch size in `neural_network.py` |

## 📚 References & Technology Stack

- **Fuzzy Logic**: scikit-fuzzy library
- **Deep Learning**: TensorFlow/Keras
- **ML Framework**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **APIs**: Google Places, Reddit (PRAW), Newspaper3k

## 📄 License

[Add your license information here]

## 👤 Authors & Contributors

[Add contributor information here]

## 📧 Support & Questions

For issues or questions, please [create an issue or contact the maintainers]
=======
# Urban Liveability Assessment Model

A fuzzy logic-based system for evaluating urban liveability based on geographic accessibility to various urban amenities and services.

## Overview

This project implements a modular fuzzy logic system to assess urban liveability using the following key features:

1. **Distance-Based Accessibility**: Calculates accessibility scores using distance decay functions
2. **Contextual Calibration**: Adjusts assessments based on urban density and development level
3. **Amenity Diversity**: Considers variety of amenities, not just proximity
4. **Fuzzy Logic Evaluation**: Handles subjective aspects of liveability with fuzzy logic rules
5. **Advanced Visualization**: Provides bar charts, radar diagrams, and comprehensive reports

## Project Structure

```
urban_bot/
├── data/
│   └── jhalwa/
│       ├── Jhalwa_Prayagraj_Transport.csv
│       ├── Jhalwa_Prayagraj_Health.csv
│       ├── Jhalwa_Prayagraj_Education.csv
│       └── ...
├── src/
│   └── fuzzy_liveability.py
├── output/
│   └── liveability_jhalwa.png
├── demo_liveability.py
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd urban_bot
   ```

2. **Install requirements**:
   ```
   pip install -r requirements.txt
   ```

3. **Run the demo**:
   ```
   python demo_liveability.py
   ```
>>>>>>> 1b4333badc55239f8451942c0e5f2c2c3f51b31a

## Data Format

The model expects data files in CSV format with the following columns:
- `Category`: The category of the amenity (e.g., Transport, Health)
- `Amenity`: The type of amenity (e.g., bus_stop, hospital)
- `Name`: Name of the specific amenity
- `Latitude`: Geographic latitude
- `Longitude`: Geographic longitude
- `Distance_km`: Distance in kilometers from the reference point
- `Travel_Time_min`: Travel time in minutes from the reference point

## Usage

### Basic Usage

```python
from src.fuzzy_liveability import analyze_location_liveability

# Analyze liveability for a location
results = analyze_location_liveability("Jhalwa")

# Access scores
overall_score = results['overall_score']
category_scores = results['category_scores']

print(f"Overall Liveability Score: {overall_score:.1f}/100")
```

### Custom Evaluation

```python
from src.fuzzy_liveability import load_location_data, evaluate_liveability, visualize_liveability_scores

# Load data
location_data = load_location_data("Jhalwa", "path/to/data")

# Evaluate liveability
results = evaluate_liveability(location_data)

# Visualize results
visualize_liveability_scores(results, "Custom Location")
```

## Advanced Configuration

You can customize the regional context parameters in `fuzzy_liveability.py`:

```python
REGIONAL_CONTEXT = {
    'urban_density': 'medium',  # Options: 'low', 'medium', 'high'
    'region_type': 'developing',  # Options: 'developing', 'developed'
    'population': 'medium',  # Options: 'low', 'medium', 'high'
    'max_reasonable_distance': 10.0  # Maximum distance (km) that's considered
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
