# STEP 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# STEP 2: LOAD THE DATASET
df = pd.read_csv("CAR DETAILS.csv")
print("✅ Dataset loaded successfully!\n")
print("🔹 First 5 rows of the dataset:")
print(df.head())
print("\n🔹 Info about the dataset:")
print(df.info())
# STEP 3: HANDLE MISSING VALUES (BASIC WAY)
df = df.dropna()
print("\n✅ After dropping missing values, shape of data:", df.shape)
# STEP 4: SEPARATE FEATURES (X) AND TARGET (y)
y = df["selling_price"]
X = df.drop(["selling_price", "name"], axis=1)
print("\n🔹 Feature columns before encoding:")
print(X.columns)
# STEP 5: CONVERT CATEGORICAL COLUMNS TO NUMBERS
X_encoded = pd.get_dummies(X, drop_first=True)
print("\n🔹 Feature columns AFTER encoding:")
print(X_encoded.columns)
# STEP 6: SPLIT DATA INTO TRAINING SET AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)
print("\n✅ Data split done!")
print("   Training data shape:", X_train.shape)
print("   Testing data shape :", X_test.shape)
# STEP 7: CREATE THE MODEL
model = RandomForestRegressor(
    n_estimators=100,   # number of trees in the forest
    random_state=42
)
# STEP 8: TRAIN (FIT) THE MODEL
print("\n⏳ Training the model...")
model.fit(X_train, y_train)
print("✅ Model training completed!")
# STEP 9: MAKE PREDICTIONS ON TEST DATA
y_pred = model.predict(X_test)
# STEP 10: EVALUATE THE MODEL
mae = mean_absolute_error(y_test, y_pred)          # average absolute error
mse = mean_squared_error(y_test, y_pred)           # average squared error
rmse = np.sqrt(mse)                                # root of MSE
r2 = r2_score(y_test, y_pred)                      # how well the model fits (1 is best)
print("\n📊 MODEL EVALUATION:")
print("   MAE  (Mean Absolute Error):", mae)
print("   RMSE (Root Mean Squared Error):", rmse)
print("   R² Score (0 to 1, higher is better):", r2)