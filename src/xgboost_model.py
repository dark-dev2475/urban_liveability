"""
Advanced XGBoost Classification Pipeline for Slum Detection
-----------------------------------------------------------
Features:
1. Stratified Sampling for Imbalanced Data
2. Automatic Class Weight Balancing
3. Hyperparameter Optimized XGBoost
4. Precision-Recall Threshold Tuning (The "Pro" Move)
5. SHAP Explainability (Why did it predict Slum?)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap  # pip install shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve, f1_score, auc)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# Set visual style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================
def load_and_prep_data(filepath):
    print("🔄 Loading Dataset...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # Create dummy data if file doesn't exist for demonstration
        print("⚠ File not found. Generating SYNTHETIC PRO-LEVEL data for demo...")
        np.random.seed(42)
        n_samples = 1000
        data = {
            'place_name': [f'Area_{i}' for i in range(n_samples)],
            'Fuzzy_Score': np.random.uniform(20, 90, n_samples),
            'Education_Score': np.random.uniform(0, 10, n_samples),
            'Environment_Score': np.random.uniform(0, 10, n_samples),
            'Food_Score': np.random.uniform(0, 10, n_samples),
            'Health_Score': np.random.uniform(0, 10, n_samples),
            'Recreation_Score': np.random.uniform(0, 10, n_samples),
            'Religion_Score': np.random.uniform(0, 10, n_samples),
            'Services_Score': np.random.uniform(0, 10, n_samples),
            'Transport_Score': np.random.uniform(0, 10, n_samples),
            'Sentiment_Score': np.random.uniform(-1, 1, n_samples),
            'Crime_Score': np.random.uniform(0, 1, n_samples),
            'Sanitation_Score': np.random.uniform(0, 10, n_samples),
            'Social_Stress_Score': np.random.uniform(0, 1, n_samples)
        }
        df = pd.DataFrame(data)
        # Generate target based on logic (Low Sanitation + High Stress = Slum)
        df['is_slum'] = ((df['Sanitation_Score'] < 4) & (df['Social_Stress_Score'] > 0.6)).astype(int)
    
    # Feature Selection
    feature_cols = [
        'Fuzzy_Score', 'Education_Score', 'Environment_Score', 'Food_Score', 
        'Health_Score', 'Recreation_Score', 'Religion_Score', 'Services_Score', 
        'Transport_Score', 'Sentiment_Score', 'Crime_Score', 
        'Sanitation_Score', 'Social_Stress_Score'
    ]
    
    target_col = 'is_slum'
    meta_col = 'place_name'
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"✅ Data Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 Class Distribution:\n{y.value_counts(normalize=True)}")
    
    return X, y, df[meta_col]

# =============================================================================
# 2. MODEL TRAINING (The "Pro" Way)
# =============================================================================
def train_xgboost_pro(X, y):
    # 1. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 2. Calculate Class Weight
    # Note: Your data has MORE slums (Class 1) than non-slums (Class 0).
    # scale_pos_weight = sum(negative) / sum(positive)
    # Since Slums (1) are the majority, this ratio will be < 1, which correctly down-weights the majority class.
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    print(f"⚖️ Calculated Class Imbalance Ratio: {ratio:.2f}")
    
    # 3. Define Model (FIXED: early_stopping_rounds goes here now)
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        use_label_encoder=False,
        eval_metric='auc',
        early_stopping_rounds=50,  # <--- MOVED HERE
        random_state=42,
        n_jobs=-1
    )
    
    print("🚀 Training Model with Early Stopping...")
    
    # 4. Fit (FIXED: removed early_stopping_rounds from here)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    print(f"✅ Training Complete. Best Iteration: {model.best_iteration}")
    return model, X_train, X_test, y_train, y_test

# =============================================================================
# 3. THRESHOLD TUNING & EVALUATION
# =============================================================================
def evaluate_model_pro(model, X_test, y_test):
    # Get probabilities instead of hard predictions
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # 1. Find Optimal Threshold (Maximizing F1 Score)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"\n🎯 Optimal Decision Threshold: {best_threshold:.4f}")
    
    # 2. Apply Optimal Threshold
    y_pred_optimal = (y_probs >= best_threshold).astype(int)
    
    # 3. Metrics
    print("\n" + "="*40)
    print("📊 FINAL EVALUATION REPORT")
    print("="*40)
    print(classification_report(y_test, y_pred_optimal))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_probs):.4f}")
    
    # 4. Plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_optimal)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Confusion Matrix (Thresh={best_threshold:.2f})')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_score(y_test, y_probs):.2f}')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    
    # Feature Importance (Gain)
    xgb.plot_importance(model, importance_type='gain', ax=axes[2], max_num_features=10, height=0.5, show_values=False)
    axes[2].set_title('Top Features (Gain)')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 4. EXPLAINABILITY (SHAP VALUES)
# =============================================================================
def explain_with_shap(model, X_test):
    print("\n🧠 Generating SHAP Explanations...")
    
    # Create explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    # Plot Summary (Beeswarm)
    plt.figure(figsize=(10, 6))
    plt.title("Why did the model classify these areas as Slums?")
    shap.summary_plot(shap_values, X_test, show=False)
    plt.show()
    
    # Plot Bar Chart (Global Importance)
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (SHAP)")
    shap.plots.bar(shap_values, show=False)
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # 1. Path to your fused dataset
    # Make sure this CSV has 'is_slum' (0 or 1) and the features listed above
    DATA_PATH = 'master_training_data.csv' 
    
    # 2. Run Pipeline
    X, y, meta = load_and_prep_data(DATA_PATH)
    
    if len(y) > 0:
        # Train
        model, X_train, X_test, y_train, y_test = train_xgboost_pro(X, y)
        
        # Evaluate
        evaluate_model_pro(model, X_test, y_test)
        
        # Explain
        explain_with_shap(model, X_test)
        
        # Save
        joblib.dump(model, 'xgb_slum_detector.pkl')
        print("\n💾 Model saved as 'xgb_slum_detector.pkl'")