import pandas as pd
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import joblib

# 1. LOADING DATA
df1 = pd.read_excel("C:/Users/kshit/OneDrive/Desktop/Internship - ML/SHM_Training_Data_LF1_JAN_MAY26.xlsx")
df = df1.drop(['HEAT_ID','LF_LAST_TEMP_TIME','CAST_START','T40_TIME'],axis = 1)


# Split features (X) and your continuous target column (y)
X = df.drop(columns=['LF2T40_TEMP_DROP']) #creates a df by dropping specified colums
y = df['LF2T40_TEMP_DROP']



# 2. SPLITTING THE DATA 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=42)

# 3. INITIALIZE & TRAIN THE REGRESSOR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


# 4. SAVING THE REGRESSOR FOR REAL-WORLD USE
joblib.dump(model, 'my_model.pkl')
print("Continuous model saved as 'my_model.pkl'")

# 5. EVALUATING THE NUMERICAL ACCURACY
import joblib
model = joblib.load('my_model.pkl')
predictions = model.predict(X_test_scaled)

# Calculating Evaluation Metrics
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2) Score: {r2 * 100}%")

# Plot Actual vs Predicted

plt.figure(figsize=(12,6))

plt.plot(y_test.values[:100], label='Actual', color='navy')
plt.plot(predictions[:100], label='Predicted', color='red', linestyle='--')

plt.xlabel('Observation Number')
plt.ylabel('Temperature Drop')
plt.title('Actual vs Predicted (First 100 Observations)')
plt.legend()
plt.grid(True)
plt.show()


