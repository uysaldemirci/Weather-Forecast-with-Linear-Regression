import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# 1. Eğitim ve test veri setlerini yükleyin
df_train = pd.read_csv("cleaned_weather_data2.csv")
df_test = pd.read_csv("cleaned_weather_data_for_prediction2.csv")

# 2. Özellikleri seçin
features = ['main.temp', 'main.feels_like', 
            'main.temp_min', 'main.temp_max', 'wind.speed', 'month', 'season', 'month_sin', 'month_cos', 'season_sin', 'season_cos']

# 3. Veriyi ölçeklendirme
scaler = MinMaxScaler()
scaled_data_train = scaler.fit_transform(df_train[features])
scaled_data_test = scaler.transform(df_test[features])

# 4. Eğitim ve test setlerini hazırlayın (son 3 gün üzerinden)
X_train, y_train = [], []
X_test, y_test = [], []

for i in range(3, len(scaled_data_train)):
    X_train.append(scaled_data_train[i-3:i].flatten())
    y_train.append(df_train['main.temp'].iloc[i])

for i in range(3, len(scaled_data_test)):
    X_test.append(scaled_data_test[i-3:i].flatten())
    y_test.append(df_test['main.temp'].iloc[i])

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# 5. GridSearchCV ile SVR Modeli
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# En iyi parametreleri alın
best_svr = grid_search.best_estimator_
print(f"En iyi parametreler: {grid_search.best_params_}")

# 6. Test verisinde tahmin yapın
test_predictions = best_svr.predict(X_test)

# 7. Performansı değerlendirin
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f"SVR Test MAE: {test_mae:.2f}")
print(f"SVR Test MSE: {test_mse:.2f}")

# 8. Gerçek ve tahmin edilen değerleri görselleştirin
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Gerçek Değerler", marker='o')
plt.plot(test_predictions, label="Tahminler", marker='x')
plt.title("SVR Modeli: Gerçek ve Tahmin Değerleri")
plt.xlabel("Veri Noktası")
plt.ylabel("Sıcaklık (°C)")
plt.legend()
plt.savefig('svr_time_series_comparison.png')
plt.show()

# 9. Tahminleri ve gerçek değerleri kaydet
results_df = pd.DataFrame({
    'True Temperature': y_test,
    'Predicted Temperature': test_predictions
})
results_df.to_csv('true_vs_predicted_temperatures_svr.csv', index=False)

print("SVR modeli tamamlandı ve sonuçlar kaydedildi.")
