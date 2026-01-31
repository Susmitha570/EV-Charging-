import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # To save trained models

# ---------------- PATH SETUP ----------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_file = os.path.join(PROJECT_ROOT, "outputs", "processed_dataset.csv")
model_dir = os.path.join(PROJECT_ROOT, "outputs", "models")
os.makedirs(model_dir, exist_ok=True)

print("Processed dataset path:", processed_file)
print("Models will be saved in:", model_dir)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(processed_file)
print("\n✅ Processed dataset loaded")
print("Shape:", df.shape)

# ---------------- DEFINE FEATURES & TARGET ----------------
target = "charging_duration_hours"

# Drop target and location columns
X = df.drop(columns=[target, "source_state", "source_city", "destination_state", "destination_city"])
y = df[target]

print("\nFeature columns:", X.columns.tolist())

# ---------------- TRAIN-TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n✅ Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# ---------------- MODEL 1: LINEAR REGRESSION ----------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Metrics (fixed RMSE calculation)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))  # ✅ manual RMSE
r2_lr = r2_score(y_test, y_pred_lr)

print("\n📌 Linear Regression Performance:")
print("MAE:", mae_lr)
print("RMSE:", rmse_lr)
print("R2 Score:", r2_lr)

# Save model
lr_model_file = os.path.join(model_dir, "linear_regression_model.pkl")
joblib.dump(lr_model, lr_model_file)
print(f"Linear Regression model saved at: {lr_model_file}")

# ---------------- MODEL 2: RANDOM FOREST REGRESSOR ----------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Metrics (fixed RMSE calculation)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))  # ✅ manual RMSE
r2_rf = r2_score(y_test, y_pred_rf)

print("\n📌 Random Forest Performance:")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)
print("R2 Score:", r2_rf)

# Save model
rf_model_file = os.path.join(model_dir, "random_forest_model.pkl")
joblib.dump(rf_model, rf_model_file)
print(f"Random Forest model saved at: {rf_model_file}")

# ---------------- SAVE PREDICTIONS ----------------
predictions = pd.DataFrame({
    "Actual": y_test,
    "Predicted_LR": y_pred_lr,
    "Predicted_RF": y_pred_rf
})
predictions_file = os.path.join(PROJECT_ROOT, "outputs", "predictions.csv")
predictions.to_csv(predictions_file, index=False)
print(f"\n✅ Predictions saved at: {predictions_file}")
