"""
Pipeline Step 7: Prediction Interface
========================================
Enter a place name → get liveable/not-liveable prediction
with confidence, feature breakdown, and explanation.
"""
import os, sys, json, numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

def load_prediction_system():
    """Load the best model, scaler, and feature info."""
    scaler_path = os.path.join(config.MODELS_DIR, "scaler.joblib")
    feature_path = os.path.join(config.MODELS_DIR, "feature_info.json")
    if not os.path.exists(scaler_path) or not os.path.exists(feature_path):
        print("❌ Model files not found. Run step_05 first.")
        return None, None, None
    scaler = joblib.load(scaler_path)
    with open(feature_path) as f:
        feature_info = json.load(f)
    feature_cols = feature_info['feature_columns']
    # Try loading models in preference order
    for model_name in ['Ensemble', 'ANN', 'XGBoost', 'RandomForest']:
        mp = os.path.join(config.MODELS_DIR, f"{model_name}.joblib")
        if os.path.exists(mp):
            model = joblib.load(mp)
            print(f"  Loaded model: {model_name}")
            return model, scaler, feature_cols
    print("❌ No trained model found.")
    return None, None, None

def get_features_for_place(place_name, df):
    """Look up features for a place from the final training CSV."""
    features = {}
    if not df.empty and place_name in df['place_name'].values:
        row = df[df['place_name'] == place_name].iloc[0]
        for col in df.columns:
            if col not in ['place_name', 'is_slum', 'is_synthetic']:
                features[col] = float(row[col])
    return features

def explain_prediction(features, prediction, probability):
    """Generate a rule-based explanation of the prediction."""
    lines = []
    if prediction == 1:
        lines.append("This area was classified as 🔴 NOT LIVEABLE (Slum-like conditions).")
        lines.append("Key negative factors:")
        if features.get('Crime_Score', 0) > 8:
            lines.append(f"  ⚠️  HIGH CRIME: Score = {features['Crime_Score']:.1f} (per-article normalized)")
        if features.get('Sanitation_Score', 0) > 5:
            lines.append(f"  ⚠️  POOR SANITATION: Score = {features['Sanitation_Score']:.1f}")
        if features.get('Social_Stress_Score', 0) > 5:
            lines.append(f"  ⚠️  SOCIAL STRESS: Score = {features['Social_Stress_Score']:.1f}")
        if features.get('Sentiment_Score', 0) < -20:
            lines.append(f"  ⚠️  NEGATIVE SENTIMENT: Score = {features['Sentiment_Score']:.1f}")
        low_amenities = [k for k in config.AMENITY_SCORE_COLUMNS if features.get(k, 0) < 3.0]
        if low_amenities:
            lines.append(f"  ⚠️  LOW AMENITY QUALITY: {', '.join(low_amenities)}")
    else:
        lines.append("This area was classified as 🟢 LIVEABLE.")
        lines.append("Positive indicators:")
        if features.get('Crime_Score', 0) < 5:
            lines.append(f"  ✅ LOW CRIME: Score = {features['Crime_Score']:.1f}")
        if features.get('Sentiment_Score', 0) > 0:
            lines.append(f"  ✅ POSITIVE SENTIMENT: Score = {features['Sentiment_Score']:.1f}")
        high_amenities = [k for k in config.AMENITY_SCORE_COLUMNS if features.get(k, 0) > 5.0]
        if high_amenities:
            lines.append(f"  ✅ GOOD AMENITY QUALITY: {', '.join(high_amenities)}")
    return "\n".join(lines)

def predict_place(place_name=None):
    """Run prediction for a place (interactive or programmatic)."""
    print("=" * 60)
    print("  🔮 Urban Liveability Prediction System")
    print("=" * 60)
    model, scaler, feature_cols = load_prediction_system()
    if model is None:
        return
    training_df = pd.read_csv(config.FINAL_TRAINING_CSV) if os.path.exists(config.FINAL_TRAINING_CSV) else pd.DataFrame()
    # Get available places
    available = set()
    if not training_df.empty:
        available.update(training_df['place_name'].tolist())
    available = sorted(list(available))
    if place_name is None:
        print(f"\nAvailable places ({len(available)}):")
        for i, p in enumerate(available):
            print(f"  {p}", end="  " if (i+1) % 4 != 0 else "\n")
        print("\n" + "-" * 40)
        place_name = input("Enter place name: ").strip()
    if place_name not in available:
        print(f"❌ '{place_name}' not found in dataset.")
        # Try fuzzy match
        matches = [p for p in available if place_name.lower() in p.lower()]
        if matches:
            print(f"   Did you mean: {', '.join(matches[:5])}?")
        return
    # Get features
    features = get_features_for_place(place_name, training_df)
    # Build feature vector in correct order
    feature_vector = np.array([[features.get(col, 0.0) for col in feature_cols]])
    feature_vector_scaled = scaler.transform(feature_vector)
    # Predict
    prediction = model.predict(feature_vector_scaled)[0]
    probability = model.predict_proba(feature_vector_scaled)[0]
    print(f"\n{'=' * 50}")
    print(f"  📍 Place: {place_name}")
    print(f"{'=' * 50}")
    print(f"\n📊 Feature Breakdown:")
    for col in feature_cols:
        val = features.get(col, 0.0)
        bar = "█" * int(val) if val >= 0 else "░" * int(abs(val) / 10)
        print(f"  {col:>25}: {val:>8.2f}  {bar}")
    print(f"\n{'=' * 50}")
    print(f"       🔮 PREDICTION 🔮")
    print(f"{'=' * 50}")
    if prediction == 1:
        print(f"\n  Result: 🔴 NOT LIVEABLE (Slum Area)")
        print(f"  Confidence: {probability[1]*100:.1f}%")
    else:
        print(f"\n  Result: 🟢 LIVEABLE")
        print(f"  Confidence: {probability[0]*100:.1f}%")
    print(f"\n--- 📝 Explanation ---")
    print(explain_prediction(features, prediction, probability))
    print(f"\n{'=' * 50}")
    return {'place': place_name, 'prediction': int(prediction), 'probability': probability.tolist(), 'features': features}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_place(sys.argv[1])
    else:
        predict_place()
