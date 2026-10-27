import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load the 9-bus dataset
df = pd.read_csv('bess_training_data_9bus.csv')

# 2. Define Features (X) and Target (y)
X = df[['load_scale', 'bus5_load_mw', 'bus8_load_mw', 'bus6_load_mw', 'tripped_line']]
y = df['required_bess_mw']

# 3. Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate accuracy
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("====================================================")
print("     9-BUS ML MODEL TRAINING RESULTS                ")
print("====================================================")
print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"R-squared Score (R2)     : {r2 * 100:.2f}% accuracy")
print("====================================================")

# 6. Save model
joblib.dump(model, 'bess_ml_controller_9bus.pkl')
print("Model saved successfully as 'bess_ml_controller_9bus.pkl'!")