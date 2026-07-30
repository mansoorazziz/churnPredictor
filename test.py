# test_load.py
import os
import joblib

# Path to your model file
model_path = os.path.join("models", "churn_model.joblib")

print("🔍 Trying to load model from:", model_path)

try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
    print("Model type:", type(model))
except Exception as e:
    print("❌ Error while loading model:")
    print(e)
