import os
import joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from dataset_loader import load_dataset

# Paths
POS_DIR = "../dataset/positives"
NEG_DIR = "../dataset/negatives"
MODEL_PATH = "../models/svm_model.pkl"

# Load data
X, y = load_dataset(POS_DIR, NEG_DIR)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM
svm = LinearSVC(max_iter=10000)
svm.fit(X_train, y_train)

# Save model
os.makedirs("../models", exist_ok=True)
joblib.dump((svm, X_test, y_test), MODEL_PATH)

print("✅ Training complete. Model saved at", MODEL_PATH)
