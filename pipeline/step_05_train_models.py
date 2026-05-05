"""
Pipeline Step 5: Train ML + DL Models
=======================================
Models trained:
  1. Random Forest (RF)
  2. SVM (RBF kernel)
  3. XGBoost
  4. ANN (Keras/TensorFlow Neural Network)
  5. Ensemble (Soft Voting)

Output: models/ directory
"""
import os, sys, json, numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.naive_bayes import GaussianNB

# XGBoost
try:
    import xgboost as xgb; HAS_XGB = True
except ImportError:
    HAS_XGB = False; print("  [WARN] XGBoost not installed")

# TensorFlow / Keras
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    HAS_KERAS = True
except ImportError:
    try:
        import keras
        from keras import layers, callbacks
        HAS_KERAS = True
    except ImportError:
        HAS_KERAS = False
        print("  [WARN] TensorFlow/Keras not installed. ANN will use sklearn MLP fallback.")

# Fallback MLP
from sklearn.neural_network import MLPClassifier


def get_feature_columns(df):
    exclude = {'place_name', 'is_slum', 'is_synthetic'}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def build_keras_ann(input_dim):
    """
    Build a proper ANN with Keras:
      Input → 128 → BN → Dropout → 64 → BN → Dropout → 32 → Output
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


class KerasANNWrapper:
    """
    Wraps Keras model to have sklearn-compatible interface
    (fit, predict, predict_proba) for VotingClassifier.
    """
    def __init__(self, input_dim, epochs=100, batch_size=16):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.model = build_keras_ann(self.input_dim)
        # Class weights for imbalance
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight = dict(zip(np.unique(y), cw))

        cb = [
            callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
        ]

        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            class_weight=class_weight,
            callbacks=cb,
            verbose=0
        )
        return self

    def predict(self, X):
        probs = self.model.predict(X, verbose=0).flatten()
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        probs = self.model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - probs, probs])

    def save(self, path):
        if not path.endswith('.keras'):
            path = path + '.keras'
        self.model.save(path)

    @classmethod
    def load(cls, path, input_dim):
        obj = cls(input_dim)
        obj.model = keras.models.load_model(path)
        return obj


def train_all_models():
    print("=" * 60)
    print("  STEP 5: Train ML + DL Models")
    print("=" * 60)

    # Load data
    for p in [config.FINAL_TRAINING_CSV, config.AUGMENTED_DATASET_CSV, config.MASTER_DATASET_CSV]:
        if os.path.exists(p): data_path = p; break
    else:
        print("  No training data found."); return None

    df = pd.read_csv(data_path)
    feature_cols = get_feature_columns(df)
    print(f"  Data: {len(df)} samples, {len(feature_cols)} features")

    X, y = df[feature_cols].values, df['is_slum'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(config.MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(config.MODELS_DIR, "feature_info.json"), 'w') as f:
        json.dump({'feature_columns': feature_cols}, f, indent=2)

    ratio = float(np.sum(y_train == 0)) / max(np.sum(y_train == 1), 1)

    # ─── Define models ───
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight='balanced', random_state=42
        ),
        'SVM_RBF': SVC(
            kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42
        ),
        'NaiveBayes': GaussianNB()
    }

    if HAS_XGB:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=ratio, eval_metric='auc',
            reg_alpha=0.1, reg_lambda=1.0, random_state=42
        )

    # ANN — Keras or sklearn MLP fallback
    if HAS_KERAS:
        models['ANN'] = KerasANNWrapper(input_dim=len(feature_cols), epochs=100, batch_size=16)
    else:
        models['ANN'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            alpha=0.001, max_iter=500, early_stopping=True,
            validation_fraction=0.15, random_state=42
        )

    # ─── Train all models ───
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        # All models use scaled data
        Xtr, Xte = X_train_s, X_test_s

        if name == 'XGBoost':
            model.fit(Xtr, y_train, verbose=False)
        else:
            model.fit(Xtr, y_train)

        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try: auc_score = roc_auc_score(y_test, y_prob)
        except: auc_score = 0.0

        results[name] = {
            'accuracy': acc, 'roc_auc': auc_score,
            'report': classification_report(y_test, y_pred, output_dict=True),
            'cm': confusion_matrix(y_test, y_pred).tolist()
        }
        trained_models[name] = model

        print(f"    Accuracy: {acc:.4f}  |  ROC-AUC: {auc_score:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Liveable', 'Slum']))

        # Save model
        if name == 'ANN' and HAS_KERAS:
            model.save(os.path.join(config.MODELS_DIR, "ANN_keras"))
            # Also save as joblib wrapper for prediction compatibility
            joblib.dump(model, os.path.join(config.MODELS_DIR, "ANN.joblib"))
        else:
            joblib.dump(model, os.path.join(config.MODELS_DIR, f"{name}.joblib"))

    # ─── Ensemble ───
    print("\n  Training Ensemble (RF + SVM + XGBoost soft vote)...")
    ensemble_estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42)),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42)),
    ]
    if HAS_XGB:
        ensemble_estimators.append(
            ('xgb', xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                       scale_pos_weight=ratio, eval_metric='auc', random_state=42))
        )

    ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    ensemble.fit(X_train_s, y_train)

    yp = ensemble.predict(X_test_s)
    ypr = ensemble.predict_proba(X_test_s)[:, 1]
    acc_e = accuracy_score(y_test, yp)
    try: auc_e = roc_auc_score(y_test, ypr)
    except: auc_e = 0.0

    results['Ensemble'] = {
        'accuracy': acc_e, 'roc_auc': auc_e,
        'report': classification_report(y_test, yp, output_dict=True),
        'cm': confusion_matrix(y_test, yp).tolist()
    }
    print(f"    Accuracy: {acc_e:.4f}  |  ROC-AUC: {auc_e:.4f}")
    print(classification_report(y_test, yp, target_names=['Liveable', 'Slum']))
    joblib.dump(ensemble, os.path.join(config.MODELS_DIR, "Ensemble.joblib"))

    # ─── Save results ───
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(os.path.join(config.PIPELINE_OUTPUT_DIR, "training_results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=conv)

    # Summary table
    print(f"\n{'='*55}")
    print(f"  {'Model':>20} | {'Accuracy':>10} | {'ROC-AUC':>10}")
    print(f"  {'-'*50}")
    for n, r in results.items():
        print(f"  {n:>20} | {r['accuracy']:>10.4f} | {r['roc_auc']:>10.4f}")
    print(f"{'='*55}")
    print(f"\n  All models saved to: {config.MODELS_DIR}")
    return results


if __name__ == "__main__":
    train_all_models()
