# EDA Analysis Module
import pickle,os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib


# make directory for models if it doesn't exist
os.makedirs("models", exist_ok=True)


# Function to load data
df = pd.read_csv('Telco-Customer-Churn.csv')

# ==================================Preprocessing and Model Training==================================
# Step 1: Handle missing values and convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Feature Selection: Define numeric and categorical columns
# Add new customer flag
df["NewCustomer"] = (df["tenure"] < 12).astype(int)
numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "NewCustomer"]
categorical_cols = ["Contract", "PaymentMethod", "InternetService","TechSupport","OnlineSecurity"]
X = df[numeric_cols + categorical_cols]
y = df["Churn"].map({"Yes": 1, "No": 0})  # Encode target



# Step 4: Preprocessing transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Step 5: Build pipeline (preprocessing + model)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    ))
])

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Fit model
model.fit(X_train, y_train)

# # Step 8: Save pipeline (includes preprocessing + model)
# pickle.dump(model, open("models/churn_model.pkl", "wb"))

#  Save after training
joblib.dump(model, "models/churn_model.joblib")
print("✅ Preprocessing + Model training complete. Model saved to models/churn_model.joblib")



# Predictions on test set
y_pred = model.predict(X_test)

# Accuracy
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Detailed report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\n🔎 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
