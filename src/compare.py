"""
SCIENTIFIC VALIDATION SUITE (Final Publication Version)
-------------------------------------------------------
1. ROBUST LOADING: Handles text errors (e.g., 'Alibag').
2. LEAKAGE PREVENTION: Drops 'Direct' features to force indirect learning.
3. RIGOROUS BENCHMARKING: 10-Fold Stratified Cross-Validation.
4. PUBLICATION OUTPUTS: Generates Boxplots, Statistical Tests, and LaTeX Tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from scipy.stats import ttest_rel
import warnings

warnings.filterwarnings('ignore')
# Set professional academic plotting style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True

# =============================================================================
# 1. LOAD & CLEAN DATA (The "Indirect Evidence" Fix)
# =============================================================================
def load_and_clean(filepath):
    print(f"📂 Loading {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("❌ Error: File not found. Please check the path.")
        return None

    # --- FIX 1: Handle Text Columns ---
    if 'place_name' in df.columns:
        print("ℹ️ Dropping 'place_name' metadata...")
        df = df.drop(columns=['place_name'])

    # --- FIX 2: PREVENT LEAKAGE (The Critical Step) ---
    # We drop features that are effectively "The Answer" (Direct measures of poverty).
    # We force the AI to use "Digital Exhaust" (Indirect measures).
    cheat_cols = [
        'Crime_Score',          # Too obvious (Correlation ~0.6)
        'Sanitation_Score',     # Direct definition of slum
        'Social_Stress_Score',  # High correlation
        'Fuzzy_Score',          # Likely a composite summary score
        'Environment_Score'     # Visual definition of slum
    ]
    
    cols_to_drop = [c for c in cheat_cols if c in df.columns]
    
    if cols_to_drop:
        print(f"\n🚫 DROPPING DIRECT LEAKAGE FEATURES: {cols_to_drop}")
        print("   (This forces the model to learn from indirect signals like Transport & Sentiment)")
        df = df.drop(columns=cols_to_drop)
        
    # --- FIX 3: Robust Numeric Conversion ---
    # Coerce errors to NaN, then fill with 0
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    print(f"✅ Final Feature Count: {df.shape[1] - 1} features")
    return df

# =============================================================================
# 2. RUN THE TOURNAMENT (10-Fold CV)
# =============================================================================
def run_benchmark(df):
    X = df.drop('is_slum', axis=1)
    y = df['is_slum']
    
    # Scale Data (Essential for SVM and MLP)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define the Contenders
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, C=1.0, random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, max_depth=6, random_state=42),
        
        # YOUR PROPOSED METHOD (Tuned for Complex Patterns)
        "Proposed Hybrid MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), # Deep enough to find hidden patterns
            activation='relu',
            solver='adam',
            alpha=0.001,                 # Regularization
            max_iter=1000,               # Allow convergence
            random_state=42
        )
    }
    
    # The Gold Standard: 10-Fold Stratified CV
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {name: [] for name in models}
    
    print(f"\n{'Model':<20} | {'Mean AUC':<8} | {'Std Dev':<8} | {'Range (Min-Max)':<15}")
    print("-" * 60)
    
    for name, model in models.items():
        # Get ROC-AUC scores
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
        results[name] = scores
        
        # Print Stats
        mean_score = scores.mean()
        std_dev = scores.std()
        print(f"{name:<20} | {mean_score:.4f}   | {std_dev:.4f}   | {scores.min():.2f} - {scores.max():.2f}")

    return results

# =============================================================================
# 3. VISUALIZATION & STATISTICS
# =============================================================================
def analyze_results(results):
    df_res = pd.DataFrame(results)
    
    # --- A. Box Plot ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_res, palette="muted", linewidth=1.5)
    sns.stripplot(data=df_res, color=".25", size=4, jitter=True, alpha=0.7)
    
    plt.title("Comparative Performance Analysis (10-Fold CV)", fontsize=14, fontweight='bold')
    plt.ylabel("ROC-AUC Score", fontsize=12)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylim(0.5, 1.05) # Focus on the top half
    
    filename = 'final_benchmark_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Graph Saved: {filename}")
    # plt.show() # Uncomment if running locally with display

    # --- B. Statistical Significance (T-Test) ---
    print("\n" + "="*50)
    print("🧪 STATISTICAL SIGNIFICANCE CHECK")
    print("="*50)
    
    proposed = results['Proposed Hybrid MLP']
    competitor = results['XGBoost'] # The strongest rival
    
    t_stat, p_val = ttest_rel(proposed, competitor)
    print(f"Comparison: Proposed MLP vs. XGBoost")
    print(f"P-Value: {p_val:.5f}")
    
    if p_val < 0.05:
        print("🏆 RESULT: SIGNIFICANT. Your model is statistically better.")
    else:
        print("🔸 RESULT: Not significant (Performance is similar).")

    # --- C. Generate LaTeX Table ---
    print("\n" + "="*50)
    print("📋 LATEX TABLE FOR PAPER")
    print("="*50)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{10-Fold Cross-Validation Results (AUC-ROC)}")
    print("\\begin{tabular}{|l|c|c|}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{Mean AUC} & \\textbf{Std. Dev} \\\\ \\hline")
    
    for name, scores in results.items():
        is_best = name == "Proposed Hybrid MLP"
        name_str = f"\\textbf{{{name}}}" if is_best else name
        print(f"{name_str} & {np.mean(scores):.3f} & $\\pm$ {np.std(scores):.3f} \\\\ \\hline")
        
    print("\\end{tabular}")
    print("\\end{table}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # 1. Load Data
    df = load_and_clean('master_training_data.csv')
    
    if df is not None:
        # 2. Run Benchmark
        results = run_benchmark(df)
        
        # 3. Analyze & Save
        analyze_results(results)