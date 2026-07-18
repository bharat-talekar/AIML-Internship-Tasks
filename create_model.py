# ==========================================
# Ford Car Price Prediction Model Training
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_excel("ford_car_dataset.xlsx")   # Change filename if needed

# ------------------------------------------
# Prepare Features and Target
# ------------------------------------------

X = df.drop("price", axis=1)
y = df["price"]

# ------------------------------------------
# One-Hot Encoding
# ------------------------------------------

X = pd.get_dummies(X)

# Save encoded column names
joblib.dump(X.columns.tolist(), "columns.pkl")

# ------------------------------------------
# Feature Scaling
# ------------------------------------------

scaler = StandardScaler()

numerical_columns = [
    "year",
    "mileage",
    "tax",
    "mpg",
    "engineSize"
]

X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# ------------------------------------------
# Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------
# Train Model
# ------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "LR_model.pkl")

print("Model Saved Successfully!")
print("Generated Files:")
print("✔ LR_model.pkl")
print("✔ scaler.pkl")
print("✔ columns.pkl")