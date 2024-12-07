import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates

# 1. Load training and testing datasets
df_train = pd.read_csv("cleaned_weather_data2.csv")
df_test = pd.read_csv("cleaned_weather_data_for_prediction2.csv")

# 2. Select features
features = ['main.temp', 'main.feels_like', 
            'main.temp_min', 'main.temp_max', 'wind.speed', 'month', 'season', 'month_sin', 'month_cos', 'season_sin', 'season_cos']

# 3. Scale the data (fit on training data, transform test data)
scaler = MinMaxScaler()
scaled_data_train = scaler.fit_transform(df_train[features])
scaled_data_test = scaler.transform(df_test[features])

# 4. Create features (X) and target (y) from training data
X_train = []
y_train = []
for i in range(3, len(scaled_data_train)):
    X_train.append(scaled_data_train[i-3:i].flatten())  # Taking the last 3 days and flattening
    y_train.append(df_train['main.temp'].iloc[i])  # Using the temperature of the 4th day as the target

X_train = np.array(X_train)
y_train = np.array(y_train)

# 5. Create features (X) and target (y) from test data
X_test = []
y_test = []
for i in range(3, len(scaled_data_test)):
    X_test.append(scaled_data_test[i-3:i].flatten())  # Taking the last 3 days and flattening
    y_test.append(df_test['main.temp'].iloc[i])  # Using the temperature of the 4th day as the target

X_test = np.array(X_test)
y_test = np.array(y_test)

# 6. Define and train the Linear Regression model
lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

# 7. Make predictions on test data
test_predictions = lr_model.predict(X_test)

# 8. Evaluate the predictions
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f"Linear Regression Test MAE: {test_mae:.2f}")
print(f"Linear Regression Test MSE: {test_mse:.2f}")

# 9. Visualize Actual vs Predicted Values over time
date_range = pd.date_range(start="2024-10-31", end="2024-11-30", freq="D")
plt.figure(figsize=(12, 6))
plt.plot(date_range, y_test, label="Actual Values", marker='o')
plt.plot(date_range, test_predictions, label="Predictions", marker='x')
plt.title("Test Data: Actual vs Predicted Values (Linear Regression)")
plt.xlabel("Date")
plt.ylabel("Temperature (K)")
plt.xticks(rotation=45)  # Rotate the date labels
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))  # Date format
plt.legend()
plt.savefig('lr_time_series_comparison.png')
plt.show()

# 10. (a) Regression Plot: Actual vs Predicted Values
plt.figure(figsize=(8, 8))
plt.scatter(y_test, test_predictions, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Line")
plt.title("Regression Plot: Actual vs Predicted (Linear Regression)")
plt.xlabel("Actual Temperature (K)")
plt.ylabel("Predicted Temperature (K)")
plt.legend()
plt.savefig('lr_regression_plot.png')
plt.show()

# 10. (b) Error Distribution Plot
errors = y_test - test_predictions
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title("Error Distribution (Linear Regression)")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.savefig('lr_error_distribution.png')
plt.show()

# 11. Save MAE and MSE Results
results = {
    "Metric": ["MAE", "MSE"],
    "Value": [test_mae, test_mse]
}

results_df = pd.DataFrame(results)
results_df.to_csv("model_performance_metrics_lr.csv", index=False)
print("MAE and MSE results 'model_performance_metrics_lr.csv' are saved.")

# Calculate Error Rate for each day (Absolute Error)
error_rate = np.abs(y_test - test_predictions)

# Save True vs Predicted Temperatures along with Error Rate
results_df = pd.DataFrame({
    'Date': date_range,
    'True Temperature': y_test,
    'Predicted Temperature': test_predictions,
    'Error Rate': error_rate
})

results_df.to_csv('true_vs_predicted_temperatures_lr.csv', index=False)

# 12. Feature Importance using Permutation Importance
from sklearn.inspection import permutation_importance

# Apply permutation importance
result = permutation_importance(lr_model, X_test, y_test, n_repeats=10, random_state=42)

# Create a DataFrame for feature importance
perm_importances = pd.DataFrame({
    'Feature': features * 3,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False)

# Print feature importance
print("Permutation Feature Importance:")
print(perm_importances)

# Plot Permutation Feature Importance
plt.figure(figsize=(12, 6))
plt.barh(perm_importances['Feature'], perm_importances['Importance'], color='green')
plt.title("Permutation Feature Importance")
plt.xlabel("Mean Decrease in Model Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()  # Reverse the order for a better visualization
plt.savefig('permutation_feature_importance.png')
plt.show()

import joblib

# Eğitilmiş modeli bir dosyaya kaydedin
joblib.dump(lr_model, 'linear_regression_model.joblib')
