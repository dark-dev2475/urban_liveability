"""
Advanced Decision Tree Classifier for Urban Liveability
-------------------------------------------------------
Features:
1. Automated Hyperparameter Tuning (Grid Search) to prevent overfitting.
2. Visualizes the actual Decision Logic (The "Tree" Diagram).
3. Handles Imbalanced Data via Class Weighting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve)
import joblib
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# =============================================================================
# 1. DATA LOADING (Same as XGBoost Pipeline)
# =============================================================================
def load_data(filepath):
    print("🔄 Loading Dataset...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("❌ File not found! Please ensure 'master_training_data.csv' exists.")
        return None, None, None
    
    # Define Features
    feature_cols = [
        'Fuzzy_Score', 'Education_Score', 'Environment_Score', 'Food_Score', 
        'Health_Score', 'Recreation_Score', 'Religion_Score', 'Services_Score', 
        'Transport_Score', 'Sentiment_Score', 'Crime_Score', 
        'Sanitation_Score', 'Social_Stress_Score'
    ]
    
    # Check if columns exist
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing columns in CSV: {missing}")
        return None, None, None

    X = df[feature_cols]
    y = df['is_slum']
    
    print(f"✅ Data Loaded: {X.shape[0]} samples")
    return X, y, feature_cols

# =============================================================================
# 2. TRAINING WITH CROSS-VALIDATION & TUNING
# =============================================================================
def train_tree_pro(X, y):
    # 1. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("🚀 Tuning Hyperparameters (Grid Search)...")
    
    # 2. Define Parameter Grid (Finds the "Sweet Spot")
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 4, 5, 7, 10],         # Controls complexity
        'min_samples_split': [2, 5, 10],       # Prevents splitting tiny groups
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']           # Crucial for Imbalance
    }
    
    # 3. Grid Search CV
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        dt, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"🏆 Best Params: {grid_search.best_params_}")
    
    return best_model, X_train, X_test, y_train, y_test

# =============================================================================
# 3. VISUALIZATION & EVALUATION
# =============================================================================
def evaluate_tree(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*40)
    print("📊 DECISION TREE EVALUATION REPORT")
    print("="*40)
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_probs):.4f}")
    
    # --- PLOT 1: Confusion Matrix & ROC ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    axes[1].plot(fpr, tpr, color='green', lw=3, label=f'AUC = {roc_auc_score(y_test, y_probs):.2f}')
    axes[1].plot([0, 1], [0, 1], linestyle='--')
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    # --- PLOT 2: The Actual Decision Tree ---
    plt.figure(figsize=(20, 10))
    plot_tree(
        model, 
        feature_names=feature_names, 
        class_names=['Non-Slum', 'Slum'],
        filled=True, 
        rounded=True, 
        fontsize=10,
        precision=2
    )
    plt.title("Visual Decision Logic (The 'Brain' of the Model)")
    plt.show()

    # --- PLOT 3: Feature Importance ---
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
    plt.title("Which Features Mattered Most?")
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    DATA_PATH = 'master_training_data.csv'  # Ensure this file exists
    
    X, y, feats = load_data(DATA_PATH)
    
    if X is not None:
        # Train
        model, X_train, X_test, y_train, y_test = train_tree_pro(X, y)
        
        # Evaluate & Visualize
        evaluate_tree(model, X_test, y_test, feats)
        
        # Save
        joblib.dump(model, 'decision_tree_slum_detector.pkl')
        print("\n💾 Model saved as 'decision_tree_slum_detector.pkl'")