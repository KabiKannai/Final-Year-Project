import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# ----------------------------
# Step 1: Load Dataset
full_df = pd.read_csv("Fraud Detection Dataset.csv")
data = full_df.head(10)  # For testing, use a subset of the data

# ----------------------------
# Step 2: Feature Engineering (both datasets)
def feature_engineering(df):
    df = df.copy()  # Ensure we're working on a copy, not a slice
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['log_amount'] = np.log1p(df['amount'])
    df['is_large_transaction'] = (df['amount'] > 10000).astype(int)
    df['same_origin_dest'] = (df['nameOrig'] == df['nameDest']).astype(int)
    df['orig_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['dest_diff'] = df['oldbalanceDest'] - df['newbalanceDest']
    df['is_sender_merchant'] = df['nameOrig'].str.startswith('M').astype(int)
    df['is_receiver_merchant'] = df['nameDest'].str.startswith('M').astype(int)
    df['repeat_flag'] = df.duplicated(subset=['nameOrig', 'nameDest']).astype(int)
    df['hour'] = df['step'] % 24
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 20)).astype(int)
    df['day'] = (df['step'] // 24).astype(int)
    df['is_weekend'] = (df['day'] % 7 >= 5).astype(int)
    return df

data = feature_engineering(data)
full_df = feature_engineering(full_df)

# ----------------------------
# Step 3: Encode 'type'
le = LabelEncoder()

# Ensure that all labels from both datasets are captured during the fitting process
le.fit(pd.concat([data['type'], full_df['type']]).unique())

# Apply the encoder to both datasets
data['type'] = le.transform(data['type'])
full_df['type'] = le.transform(full_df['type'])

# ----------------------------
# Step 4: Prepare Features and Target
X = data.drop(columns=['isFraud', 'nameOrig', 'nameDest'])
y = data['isFraud']
X_numeric = X.select_dtypes(include=[np.number])

# Handle NaNs
X_numeric = X_numeric.fillna(0)

# ----------------------------
# Step 5: Mutual Information Feature Selection (handles negative values)
mi_selector = SelectKBest(mutual_info_classif, k='all')
mi_selector.fit(X_numeric, y)
significant_features = X_numeric.columns[mi_selector.get_support()]
X_selected = X_numeric[significant_features]

# Visualize Feature Importance
scores = mi_selector.scores_
plt.figure(figsize=(10, 6))
sns.barplot(x=scores, y=X_numeric.columns)
plt.title("Mutual Information Feature Scores")
plt.xlabel("Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ----------------------------
# Step 6: Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# ----------------------------
# Step 7: SMOTE
if y.sum() >= 2:
    smote = SMOTE(k_neighbors=min(5, y.sum()-1), random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
else:
    X_resampled, y_resampled = X_scaled, y

# Visualize SMOTE result
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_resampled[:, 0], y=X_resampled[:, 1], hue=y_resampled, palette='coolwarm')
plt.title("SMOTE Resampled Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# ----------------------------
# Step 8: PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_resampled)

# Visualize PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_resampled, palette='coolwarm')
plt.title("PCA Projection of Resampled Data")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

# ----------------------------
# Step 9: Chi-Square Test for Feature Selection
# Ensure features are non-negative before applying Chi-Square
X_scaled_non_negative = np.abs(X_scaled)  # Apply absolute value to ensure non-negative data
chi2_selector = SelectKBest(chi2, k='all')
chi2_selector.fit(X_scaled_non_negative, y)
chi2_scores = chi2_selector.scores_

# Visualize Chi-Square Feature Scores
plt.figure(figsize=(10, 6))
sns.barplot(x=chi2_scores, y=X_selected.columns)
plt.title("Chi-Square Feature Scores")
plt.xlabel("Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ----------------------------
# Step 10: Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_pca, y_resampled)

# ----------------------------
# Step 11: Predict on First 10
X_orig_scaled = scaler.transform(X_selected)
X_orig_pca = pca.transform(X_orig_scaled)
data['predicted'] = model.predict(X_orig_pca)
print("\nFinal Predictions on First 10 Rows:\n")
print(data[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'isFraud', 'predicted']])

# ----------------------------
# Step 12: Apply to Full Dataset
X_full = full_df[significant_features].fillna(0)
X_full_scaled = scaler.transform(X_full)
X_full_pca = pca.transform(X_full_scaled)
full_df['predicted'] = model.predict(X_full_pca)

# ----------------------------
# Step 13: Top 10 Fraud Accounts
fraud_pred = full_df[full_df['predicted'] == 1]
fraud_counts = fraud_pred['nameOrig'].value_counts().reset_index()
fraud_counts.columns = ['account', 'fraud_count']
total_counts = full_df['nameOrig'].value_counts().reset_index()
total_counts.columns = ['account', 'total_count']
fraud_stats = pd.merge(fraud_counts, total_counts, on='account')
fraud_stats['fraud_rate'] = (fraud_stats['fraud_count'] / fraud_stats['total_count']) * 100
top_10_fraud_accounts = fraud_stats.sort_values(by='fraud_count', ascending=False).head(10)

print("\nTop 10 Predicted Fraud Accounts with Fraud Rate:\n")
print(top_10_fraud_accounts[['account', 'fraud_count', 'total_count', 'fraud_rate']])
