import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras import activations
from keras.callbacks import ReduceLROnPlateau


# 1. Eğitim ve test veri setlerini yükleyin
df_train = pd.read_csv("cleaned_weather_data2.csv")
df_test = pd.read_csv("cleaned_weather_data_for_prediction2.csv")

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

# # 5. Modeli tanımlayın
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1)  # Çıkış katmanı (sıcaklık tahmini)
])

# model = Sequential([
#     Dense(64, input_dim=X_train.shape[1], activation=activations.swish),
#     Dense(32, activation=activations.swish),
#     Dense(1)  # Çıkış katmanı (sıcaklık tahmini)
# ])

lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-6)

# model = Sequential([
#     Dense(256, activation='swish', input_dim=X_train.shape[1]),
#     Dense(128, activation='swish'),
#     Dense(64, activation='swish'),
#     Dense(32, activation='swish'),
#     Dense(16, activation='swish'),
#     Dense(1)
# ])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
# 6. EarlyStopping ile modeli eğitin
early_stopping = EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)

model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1, callbacks=[early_stopping, lr_scheduler])
# model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=1, callbacks=[early_stopping])

# 7. Test verisinden özellikleri (X) ve hedefi (y) oluşturun
X_test = []
y_test = []
for i in range(3, len(scaled_data_test)):
    X_test.append(scaled_data_test[i-3:i].flatten())  # Son 3 günü alıp düzleştiriyoruz
    y_test.append(df_test['main.temp'].iloc[i])  # 4. günün sıcaklığını hedef olarak alıyoruz

X_test = np.array(X_test)
y_test = np.array(y_test)

# 8. Test verisi üzerinde tahmin yapın
test_predictions = model.predict(X_test)

# 9. Tahminleri değerlendirin
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f"Test MAE: {test_mae:.2f}")
print(f"Test MSE: {test_mse:.2f}")

print(y_test.shape)  # ytest'in boyutunu kontrol et
print(test_predictions.shape)  # test_predictions'in boyutunu kontrol et
test_predictions = test_predictions.flatten()
print(test_predictions.shape)

# Tahminleri ve gerçek değerleri kaydet
results_df = pd.DataFrame({
    'True Temperature': y_test,
    'Predicted Temperature': test_predictions
})
results_df.to_csv('true_vs_predicted_temperatures_ann.csv', index=False)

# 10. Görselleştirme ve grafik kaydetme
# (a) Zaman Serisi Grafiği
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Gerçek Değerler", marker='o')
plt.plot(test_predictions, label="Tahminler", marker='x')
plt.title("Test Verisi: Gerçek ve Tahmin Edilen Değerler")
plt.xlabel("Veri Noktası")
plt.ylabel("Sıcaklık (°C)")
plt.legend()
plt.savefig('time_series_comparison_ann.png')
plt.show()

# (b) Regresyon Grafiği (Gerçek vs Tahmin)
plt.figure(figsize=(8, 8))
plt.scatter(y_test, test_predictions, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Doğru Çizgi")
plt.title("Regresyon Grafiği: Gerçek vs Tahmin")
plt.xlabel("Gerçek Sıcaklık (°C)")
plt.ylabel("Tahmin Edilen Sıcaklık (°C)")
plt.legend()
plt.savefig('regression_plot_ann.png')
plt.show()

# (c) Hata Dağılım Grafiği
errors = y_test - test_predictions
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title("Hata Dağılımı")
plt.xlabel("Hata (Gerçek - Tahmin)")
plt.ylabel("Frekans")
plt.savefig('error_distribution_ann.png')
plt.show()

# 11. MAE ve MSE Sonuçlarını Kaydet
results = {
    "Metric": ["MAE", "MSE"],
    "Value": [test_mae, test_mse]
}

results_df = pd.DataFrame(results)
results_df.to_csv("model_performance_metrics_ann.csv", index=False)
print("MAE ve MSE sonuçları 'model_performance_metrics_ann.csv' dosyasına kaydedildi.")