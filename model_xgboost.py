import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# 6. XGBoost Modelini Tanımlayın
xgb_model = XGBRegressor(
    n_estimators=200,  # Ağaç sayısı
    learning_rate=0.05,  # Öğrenme oranı
    max_depth=6,  # Maksimum derinlik
    subsample=0.8,  # Alt örnekleme oranı
    colsample_bytree=0.8,  # Özellik alt örnekleme oranı
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 7. Test Verisinde Tahmin Yapın
test_predictions = xgb_model.predict(X_test)

# 8. Tahminleri Değerlendirin
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f"XGBoost Test MAE: {test_mae:.2f}")
print(f"XGBoost Test MSE: {test_mse:.2f}")

# 9. Tahminleri ve Gerçek Değerleri Görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Gerçek Değerler", marker='o')
plt.plot(test_predictions, label="Tahminler", marker='x')
plt.title("Test Verisi: Gerçek ve Tahmin Edilen Değerler (XGBoost)")
plt.xlabel("Veri Noktası")
plt.ylabel("Sıcaklık (°C)")
plt.legend()
plt.savefig('xgb_time_series_comparison.png')
plt.show()

# 10. Regresyon ve Hata Dağılımı Grafikleri
# (a) Regresyon Grafiği
plt.figure(figsize=(8, 8))
plt.scatter(y_test, test_predictions, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Doğru Çizgi")
plt.title("Regresyon Grafiği: Gerçek vs Tahmin (XGBoost)")
plt.xlabel("Gerçek Sıcaklık (°C)")
plt.ylabel("Tahmin Edilen Sıcaklık (°C)")
plt.legend()
plt.savefig('xgb_regression_plot.png')
plt.show()

# (b) Hata Dağılımı Grafiği
errors = y_test - test_predictions
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title("Hata Dağılımı (XGBoost)")
plt.xlabel("Hata (Gerçek - Tahmin)")
plt.ylabel("Frekans")
plt.savefig('xgb_error_distribution.png')
plt.show()

print(y_test.shape)  # ytest'in boyutunu kontrol et
print(test_predictions.shape)  # test_predictions'in boyutunu kontrol et
test_predictions = test_predictions.flatten()
print(test_predictions.shape)

# Tahminleri ve gerçek değerleri kaydet
results_df = pd.DataFrame({
    'True Temperature': y_test,
    'Predicted Temperature': test_predictions
})
results_df.to_csv('true_vs_predicted_temperatures_xgboost.csv', index=False)

# 11. MAE ve MSE Sonuçlarını Kaydet
results = {
    "Metric": ["MAE", "MSE"],
    "Value": [test_mae, test_mse]
}

results_df = pd.DataFrame(results)
results_df.to_csv("model_performance_metrics_xgboost.csv", index=False)
print("MAE ve MSE sonuçları 'model_performance_metrics_xgboost.csv' dosyasına kaydedildi.")