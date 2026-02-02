import joblib
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "../models/svm_model.pkl"

svm, X_test, y_test = joblib.load(MODEL_PATH)

y_pred = svm.predict(X_test)

print("📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
