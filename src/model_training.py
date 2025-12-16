from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import (
    load_data,
    split_features_target, preprocess_features,
    )


DATA_PATH = Path("data/loanData.csv")


def main():
  
  
    # Train / Test split
    
    df = load_data(DATA_PATH)

    print("Creating Risk Ratios...")
    df['loan_to_income'] = df['loan_amount'] / df['income_annum']
    total_assets = df['residential_assets_value'] + df['commercial_assets_value'] + df['luxury_assets_value'] + df['bank_asset_value']
    df['asset_to_loan'] = total_assets / df['loan_amount']

    #print("Dropping 'cibil_score' to test financial fundamentals...")
    # if 'cibil_score' in df.columns:
      #  df = df.drop(columns=['cibil_score'])
    
   
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    # prepare data
    X_train = preprocess_features(X_train)
    X_test = preprocess_features(X_test)
   
    # Scale features and handle class imbalance with SMOTE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    

    # Logistic Regression (Baseline)
    lr_model = LogisticRegression(max_iter=1000, solver="liblinear")
    lr_model.fit(X_train_resampled, y_train_resampled)

    y_pred_lr = lr_model.predict(X_test_scaled)
    y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

    accuracy_lr = accuracy_score(y_test, y_pred_lr) 
    roc_lr = roc_auc_score(y_test, y_proba_lr) 

 
    # Decision Tree
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
    dt_model.fit(X_train_resampled, y_train_resampled)
    feature_names = X_train.columns
    importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': dt_model.feature_importances_
    })
    y_pred_dt = dt_model.predict(X_test_scaled)
    y_proba_dt = dt_model.predict_proba(X_test_scaled)[:, 1]

    accuracy_dt = accuracy_score(y_test, y_pred_dt) 
    roc_dt = roc_auc_score(y_test, y_proba_dt) 

   
   # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    rf_model.fit(X_train_resampled, y_train_resampled)
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

    acc_rf = accuracy_score(y_test, y_pred_rf) 
    roc_rf = roc_auc_score(y_test, y_proba_rf) 
   
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    })
   
    # Print Results
    print("\n" + "="*40)
    print("MODEL COMPARISON RESULTS")
    print("="*40)

    print("\nTop 5 Features (Random Forest):")
    print(importances.sort_values(by='Importance', ascending=False).head(5))

    print("\nPerformance Metrics:")
    print("-" * 30)
    print(f"Logistic Regression: Accuracy={accuracy_lr:.4f} | AUC={roc_lr:.4f}")
    print(f"Decision Tree:       Accuracy={accuracy_dt:.4f} | AUC={roc_dt:.4f}")
    print(f"Random Forest:       Accuracy={acc_rf:.4f} | AUC={roc_rf:.4f}")
    
    print("\nRandom Forest detailed Report:")
    print(classification_report(y_test, y_pred_rf))

    # Confusion Matrix for Random Forest
    cm = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Approved', 'Rejected'],
                yticklabels=['Approved', 'Rejected'])
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    main()

