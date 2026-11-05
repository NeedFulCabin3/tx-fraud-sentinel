import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def load_or_generate_data():
  """Generates a synthetic imbalanced dataset simulating credit card transactions.

  Class 0: Legitimate (99%), Class 1: Fraudulent (1%)
  """
  print('Generating synthetic transaction dataset...')
  X, y = make_classification(
      n_samples=20000,
      n_features=12,
      n_informative=8,
      n_redundant=4,
      weights=[0.99, 0.01],  # Severe class imbalance
      random_state=42,
  )

  # Convert to a DataFrame to mimic a real tabular dataset
  feature_names = [f'V{i}' for i in range(1, 11)] + ['Amount', 'Time']
  df = pd.DataFrame(X, columns=feature_names)
  # Scale 'Amount' and 'Time' to look more realistic
  df['Amount'] = np.abs(df['Amount']) * 100
  df['Time'] = np.abs(df['Time']) * 3600

  return df, y


def build_pipeline():
  """Constructs the scikit-learn Pipeline with RobustScaler and RandomForest."""
  return Pipeline([
      (
          'scaler',
          RobustScaler(),
      ),  # Less sensitive to outliers common in financial data
      (
          'classifier',
          RandomForestClassifier(
              n_estimators=150,
              max_depth=10,
              class_weight='balanced',  # Handles class imbalance natively
              random_state=42,
              n_jobs=-1,
          ),
      ),
  ])


def optimize_threshold(y_true, y_scores):
  """Finds the optimal decision threshold using precision_recall_curve

  to maximize the F1-score.
  """
  precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

  # Handle division by zero safely
  f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
  best_idx = np.argmax(f1_scores)

  # precision_recall_curve returns one more precision/recall value than thresholds
  best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
  best_f1 = f1_scores[best_idx]

  return best_threshold, best_f1


def main():
  try:
    # 1. Load Data
    X, y = load_or_generate_data()

    # 2. Train-Test Split (stratified to preserve class ratio)
    print('Splitting data into train and test sets...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # 3. Build and Train Pipeline
    print('Training Random Forest Pipeline...')
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 4. Predict Probabilities
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # 5. Evaluate ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f'\n[Evaluation] ROC-AUC Score: {roc_auc:.4f}')

    # 6. Tune Decision Threshold
    best_threshold, best_f1 = optimize_threshold(y_test, y_pred_proba)
    print(f'Optimal Threshold (Max F1): {best_threshold:.4f} (F1: {best_f1:.4f})')

    # Evaluate with default threshold (0.5) vs tuned threshold
    print('\n--- Classification Report (Default Threshold: 0.5) ---')
    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    print(classification_report(y_test, y_pred_default, zero_division=0))

    print(
        f'--- Classification Report (Tuned Threshold: {best_threshold:.4f}) ---'
    )
    y_pred_tuned = (y_pred_proba >= best_threshold).astype(int)
    print(classification_report(y_test, y_pred_tuned, zero_division=0))

    # 7. Interactive Terminal Prediction Prompt
    print('\n' + '=' * 40)
    print('Interactive Fraud Probability Tester')
    print('=' * 40)
    while True:
      user_input = input(
          "Enter a mock fraud probability score (0.0 to 1.0) to evaluate (or type"
          " 'exit'): "
      ).strip()
      if user_input.lower() == 'exit':
        print('Exiting pipeline script. Stay secure!')
        break

      try:
        score = float(user_input)
        if not (0.0 <= score <= 1.0):
          print('Please enter a valid probability value between 0.0 and 1.0.')
          continue

        is_fraud_default = score >= 0.5
        is_fraud_tuned = score >= best_threshold

        print(f'  -> Probability: {score:.4f}')
        print(
            f'  -> Flagged at Default (0.5):'
            f' {"FRAUD 🚨" if is_fraud_default else "Legitimate ✅"}'
        )
        print(
            f'  -> Flagged at Tuned ({best_threshold:.4f}):'
            f' {"FRAUD 🚨" if is_fraud_tuned else "Legitimate ✅"}'
        )
        print('-' * 40)
      except ValueError:
        print('Invalid input. Please enter a numerical float or type "exit".')

  except Exception as e:
    print(f'An error occurred during pipeline execution: {e}')


if __name__ == '__main__':
  main()