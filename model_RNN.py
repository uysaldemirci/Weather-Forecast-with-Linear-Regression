import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

# 1. Eğitim ve test veri setlerini yükleyin
df_train = pd.read_csv("cleaned_weather_data.csv")
df_test = pd.read_csv("cleaned_weather_data_for_prediction.csv")

# 2. Özellikleri seçin
features = ['main.temp', 'main.feels_like', 'main.pressure', 'main.humidity', 
            'main.temp_min', 'main.temp_max', 'wind.speed', 'wind.deg', 'clouds.all']

# 3. Veriyi ölçeklendirme
scaler = MinMaxScaler()
scaled_data_train = scaler.fit_transform(df_train[features])
scaled_data_test = scaler.transform(df_test[features])

# 4. Eğitim verisinden özellikleri (X) ve hedefi (y) oluşturun
X_train = []
y_train = []
for i in range(3, len(scaled_data_train)):
    X_train.append(scaled_data_train[i-3:i])  # Son 3 gün (zaman serisi) olarak alıyoruz
    y_train.append(df_train['main.temp'].iloc[i])  # 4. günün sıcaklığını hedef olarak alıyoruz

X_train = np.array(X_train)  # (örnek sayısı, zaman adımı, özellik sayısı)
y_train = np.array(y_train)

# 5. Test verisinden özellikleri (X) ve hedefi (y) oluşturun
X_test = []
y_test = []
for i in range(3, len(scaled_data_test)):
    X_test.append(scaled_data_test[i-3:i])  # Son 3 gün (zaman serisi)
    y_test.append(df_test['main.temp'].iloc[i])  # 4. günün sıcaklığı

X_test = np.array(X_test)
y_test = np.array(y_test)

# 6. RNN Modelini tanımlayın
model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(32, activation='relu'),
    Dense(1)  # Çıkış katmanı: Sıcaklık tahmini
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 7. Modeli eğitin
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 8. Test verisi üzerinde tahmin yapın
test_predictions = model.predict(X_test)

# 9. Tahminleri değerlendirin
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f"Test MAE: {test_mae:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# 10. Gerçek ve Tahmin Edilen Sıcaklıkları karşılaştırın
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Gerçek Değerler", marker='o')
plt.plot(test_predictions, label="Tahminler", marker='x')
plt.title("Test Verisi: Gerçek ve Tahmin Edilen Değerler")
plt.xlabel("Veri Noktası")
plt.ylabel("Sıcaklık (°C)")
plt.legend()
plt.show()

# 11. Tahmin sonuçlarını kaydet
results_df = pd.DataFrame({
    'True Temperature': y_test,
    'Predicted Temperature': test_predictions.flatten()
})
results_df.to_csv('true_vs_predicted_temperatures_rnn.csv', index=False)

