import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# 1. Eğitim ve test veri setlerini yükleyin
df_train = pd.read_csv("cleaned_weather_data.csv")
df_test = pd.read_csv("cleaned_weather_data_for_prediction.csv")

# 2. Özellikleri seçin
features = ['main.temp', 'main.feels_like', 'main.pressure', 'main.humidity', 
            'main.temp_min', 'main.temp_max', 'wind.speed', 'wind.deg', 'clouds.all']

# 3. Veriyi ölçeklendirme (eğitim verisiyle fit, test verisine transform uygulanıyor)
scaler = MinMaxScaler()
scaled_data_train = scaler.fit_transform(df_train[features])
scaled_data_test = scaler.transform(df_test[features])

# 4. Eğitim verisinden özellikleri (X) ve hedefi (y) oluşturun
X_train = []
y_train = []
for i in range(3, len(scaled_data_train)):
    X_train.append(scaled_data_train[i-3:i].flatten())  # Son 3 günü alıp düzleştiriyoruz
    y_train.append(df_train['main.temp'].iloc[i])  # 4. günün sıcaklığını hedef olarak alıyoruz

X_train = np.array(X_train)
y_train = np.array(y_train)

# 5. Test verisinden özellikleri (X) ve hedefi (y) oluşturun
X_test = []
y_test = []
for i in range(3, len(scaled_data_test)):
    X_test.append(scaled_data_test[i-3:i].flatten())  # Son 3 günü alıp düzleştiriyoruz
    y_test.append(df_test['main.temp'].iloc[i])  # 4. günün sıcaklığını hedef olarak alıyoruz

X_test = np.array(X_test)
y_test = np.array(y_test)

# Gradient Boosting Regressor modelini tanımla
gbr_model = GradientBoostingRegressor(random_state=42)

# Modeli eğit
gbr_model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yap
test_predictions = gbr_model.predict(X_test)

# Modelin performansını değerlendirme
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f"Test MAE: {test_mae:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Parametre grid'ini tanımla
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV ile modelin hiperparametrelerini optimize et
grid_search = GridSearchCV(estimator=gbr_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# En iyi parametreleri yazdır
print(f"En iyi parametreler: {grid_search.best_params_}")

# En iyi modeli kullan
best_gbr_model = grid_search.best_estimator_

# En iyi modeli al
best_gbr_model = grid_search.best_estimator_

# Test verisi üzerinde tahmin yap
best_test_predictions = best_gbr_model.predict(X_test)

# Modelin performansını değerlendirme
best_test_mae = mean_absolute_error(y_test, best_test_predictions)
best_test_mse = mean_squared_error(y_test, best_test_predictions)

print(f"Best Test MAE: {best_test_mae:.2f}")
print(f"Best Test MSE: {best_test_mse:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Gerçek Değerler", marker='o')
plt.plot(best_test_predictions, label="Tahminler", marker='x')
plt.title("Gerçek ve Tahmin Edilen Değerler (Gradient Boosting)")
plt.xlabel("Veri Noktası")
plt.ylabel("Sıcaklık (°C)")
plt.legend()
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_test, best_test_predictions, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Doğru Çizgi")
plt.title("Regresyon Grafiği: Gerçek vs Tahmin (Gradient Boosting)")
plt.xlabel("Gerçek Sıcaklık (°C)")
plt.ylabel("Tahmin Edilen Sıcaklık (°C)")
plt.legend()
plt.show()

errors = y_test - best_test_predictions
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title("Hata Dağılımı (Gradient Boosting)")
plt.xlabel("Hata (Gerçek - Tahmin)")
plt.ylabel("Frekans")
plt.show()

