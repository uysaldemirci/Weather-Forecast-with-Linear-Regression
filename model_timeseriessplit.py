import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# 1. Eğitim ve test veri setlerini yükleyin
df_train = pd.read_csv("cleaned_weather_data.csv")
df_test = pd.read_csv("cleaned_weather_data_for_prediction.csv")

# 2. Özellikleri seçin
features = ['main.temp', 'main.feels_like', 'main.pressure', 'main.humidity', 
            'main.temp_min', 'main.temp_max', 'wind.speed', 'wind.deg', 'clouds.all']

# 3. Veriyi ölçeklendiriyoruz
scaler = MinMaxScaler()
scaled_data_train = scaler.fit_transform(df_train[features])

# 4. Özellikleri (X) ve hedefi (y) oluşturun
X = []
y = []
for i in range(3, len(scaled_data_train)):
    X.append(scaled_data_train[i-3:i].flatten())  # Son 3 günü alıp düzleştiriyoruz
    y.append(df_train['main.temp'].iloc[i])  # 4. günün sıcaklığını hedef olarak alıyoruz

X = np.array(X)
y = np.array(y)

# 5. TimeSeriesSplit için eğitim verisini bölme
tscv = TimeSeriesSplit(n_splits=5)

mae_scores = []
mse_scores = []

# 6. Modeli tanımlıyoruz
model = Sequential([
    Dense(64, activation='relu', input_dim=X.shape[1]),
    Dense(32, activation='relu'),
    Dense(1)  # Çıkış katmanı (sıcaklık tahmini)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# 7. TimeSeriesSplit ile her bölüm için eğitim ve doğrulama işlemi
for train_index, val_index in tscv.split(X):
    # Eğitim ve doğrulama verilerini ayırıyoruz
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 8. Modeli eğitiyoruz
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    # 9. Doğrulama verisi üzerinde tahmin yapıyoruz
    val_predictions = model.predict(X_val)

    # 10. Doğrulama hatalarını hesaplıyoruz
    val_mae = mean_absolute_error(y_val, val_predictions)
    val_mse = mean_squared_error(y_val, val_predictions)

    mae_scores.append(val_mae)
    mse_scores.append(val_mse)

# 11. Ortalama MSE ve MAE'yi yazdırıyoruz
print(f'Mean MAE: {np.mean(mae_scores)}')
print(f'Mean MSE: {np.mean(mse_scores)}')

# 12. Modeli test verisi ile değerlendiriyoruz
test_features = df_test[features]
scaled_data_test = scaler.transform(test_features)

X_test = []
y_test = []
for i in range(3, len(scaled_data_test)):
    X_test.append(scaled_data_test[i-3:i].flatten())  # Son 3 günü alıyoruz
    y_test.append(df_test['main.temp'].iloc[i])  # 4. günün sıcaklığını hedef olarak alıyoruz

X_test = np.array(X_test)
y_test = np.array(y_test)

# Test verisi üzerinde tahmin yapıyoruz
test_predictions = model.predict(X_test)

# Test hatalarını hesaplıyoruz
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f'Test MAE: {test_mae}')
print(f'Test MSE: {test_mse}')

print(y_test.shape)  # ytest'in boyutunu kontrol et
print(test_predictions.shape)  # test_predictions'in boyutunu kontrol et
test_predictions = test_predictions.flatten()
print(test_predictions.shape)

# 10. Görselleştirme ve grafik kaydetme
# (a) Zaman Serisi Grafiği
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Gerçek Değerler", marker='o')
plt.plot(test_predictions, label="Tahminler", marker='x')
plt.title("Test Verisi: Gerçek ve Tahmin Edilen Değerler")
plt.xlabel("Veri Noktası")
plt.ylabel("Sıcaklık (°C)")
plt.legend()
plt.savefig('time_series_comparison.png')
plt.show()

# (b) Regresyon Grafiği (Gerçek vs Tahmin)
plt.figure(figsize=(8, 8))
plt.scatter(y_test, test_predictions, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Doğru Çizgi")
plt.title("Regresyon Grafiği: Gerçek vs Tahmin")
plt.xlabel("Gerçek Sıcaklık (°C)")
plt.ylabel("Tahmin Edilen Sıcaklık (°C)")
plt.legend()
plt.savefig('regression_plot.png')
plt.show()

# (c) Hata Dağılım Grafiği
errors = y_test - test_predictions
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title("Hata Dağılımı")
plt.xlabel("Hata (Gerçek - Tahmin)")
plt.ylabel("Frekans")
plt.savefig('error_distribution.png')
plt.show()




results_df = pd.DataFrame({
    'True Temperature': y_test,
    'Predicted Temperature': test_predictions
})

results_df.to_csv('true_vs_predicted_temperatures.csv', index=False)