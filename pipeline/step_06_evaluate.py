"""
Pipeline Step 6: Evaluate & Compare Models
=============================================
10-Fold Stratified CV, ROC curves, confusion matrices,
SHAP analysis, boxplots, LaTeX tables.
"""
import os, sys, json, numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from scipy.stats import ttest_rel
from sklearn.naive_bayes import GaussianNB
try:
    import xgboost as xgb; HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import shap; HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

def get_feature_columns(df):
    exclude = {'place_name', 'is_slum', 'is_synthetic'}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def run_evaluation():
    print("=" * 60)
    print("  STEP 6: Evaluate & Compare Models (10-Fold CV)")
    print("=" * 60)
    for p in [config.FINAL_TRAINING_CSV, config.AUGMENTED_DATASET_CSV, config.MASTER_DATASET_CSV]:
        if os.path.exists(p): data_path = p; break
    else:
        print("No data found."); return
    df = pd.read_csv(data_path)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values; y = df['is_slum'].values
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    ratio = float(np.sum(y==0)) / max(np.sum(y==1), 1)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        'ANN (Proposed)': MLPClassifier(hidden_layer_sizes=(128,64,32), activation='relu', alpha=0.001, max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=42),
        'Naive Bayes': GaussianNB(),
    }
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, scale_pos_weight=ratio, eval_metric='auc', random_state=42)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)
    cv_results = {}
    print(f"\n{'Model':<20} | {'Mean AUC':>10} | {'Std':>8} | {'Min-Max':>12}")
    print("-" * 60)
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
        cv_results[name] = scores
        print(f"{name:<20} | {scores.mean():>10.4f} | {scores.std():>8.4f} | {scores.min():.2f}-{scores.max():.2f}")
    # Boxplot
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    df_res = pd.DataFrame(cv_results)
    sns.boxplot(data=df_res, palette='muted', linewidth=1.5)
    sns.stripplot(data=df_res, color='.25', size=4, jitter=True, alpha=0.7)
    plt.title('Model Comparison (10-Fold Stratified CV)', fontsize=14, fontweight='bold')
    plt.ylabel('ROC-AUC Score', fontsize=12)
    plt.ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'cv_comparison_boxplot.png'), dpi=300)
    plt.close()
    print(f"\n  Saved: cv_comparison_boxplot.png")
    # ROC Curves
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.3f})')
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'roc_curves.png'), dpi=300)
    plt.close()
    print(f"  Saved: roc_curves.png")
    # Confusion matrices
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
    if len(models) == 1: axes = [axes]
    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X_tr, y_tr)
        yp = model.predict(X_te)
        cm = confusion_matrix(y_te, yp)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Liveable','Slum'], yticklabels=['Liveable','Slum'])
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'confusion_matrices.png'), dpi=300)
    plt.close()
    print(f"  Saved: confusion_matrices.png")
    # Statistical significance
    print(f"\n  Statistical Significance (Paired t-test vs Proposed MLP):")
    if 'ANN (Proposed)' in cv_results:
        proposed = cv_results['ANN (Proposed)']
        for name, scores in cv_results.items():
            if name == 'ANN (Proposed)': continue
            t_stat, p_val = ttest_rel(proposed, scores)
            sig = "SIGNIFICANT" if p_val < 0.05 else "Not significant"
            print(f"    vs {name}: p={p_val:.5f} -> {sig}")
    # SHAP (on best model)
    if HAS_SHAP and HAS_XGB and 'XGBoost' in models:
        print(f"\n  Generating SHAP explanations...")
        try:
            xgb_model = models['XGBoost']
            xgb_model.fit(X_tr, y_tr)
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_te)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_te, feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(config.FIGURES_DIR, 'shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_te, feature_names=feature_cols, plot_type='bar', show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(config.FIGURES_DIR, 'shap_bar.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: shap_summary.png, shap_bar.png")
        except Exception as e:
            print(f"  SHAP skipped due to: {e}")
    # LaTeX table
    print(f"\n  LaTeX Table:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{10-Fold Cross-Validation Results (ROC-AUC)}")
    print("\\begin{tabular}{|l|c|c|}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{Mean AUC} & \\textbf{Std Dev} \\\\ \\hline")
    for name, scores in cv_results.items():
        bold = "\\textbf{" + name + "}" if name == "ANN (Proposed)" else name
        print(f"{bold} & {scores.mean():.3f} & $\\pm$ {scores.std():.3f} \\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    # Save CV results
    cv_summary = {n: {'mean': float(s.mean()), 'std': float(s.std()), 'scores': s.tolist()} for n, s in cv_results.items()}
    with open(os.path.join(config.PIPELINE_OUTPUT_DIR, "cv_results.json"), 'w') as f:
        json.dump(cv_summary, f, indent=2)
    print(f"\n✅ Evaluation complete. Figures saved to {config.FIGURES_DIR}")

if __name__ == "__main__":
    run_evaluation()
