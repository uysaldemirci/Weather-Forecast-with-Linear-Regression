import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# 1. Eğitim ve test veri setlerini yükleyin
df_train = pd.read_csv("cleaned_weather_data2.csv")
df_test = pd.read_csv("cleaned_weather_data_for_prediction2.csv")

# 2. Özellikleri seçin
features = ['main.temp', 'main.feels_like', 'main.pressure', 'main.humidity', 
            'main.temp_min', 'main.temp_max', 'wind.speed', 'wind.deg', 'clouds.all', 'month', 'season', 'month_sin', 'month_cos', 'season_sin', 'season_cos']

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

# param_grid = {
#     'n_estimators': [100, 200, 300, 500],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2']
# }

# # GridSearchCV ile modeli oluştur
# grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
#                            param_grid=param_grid, 
#                            cv=5,   # K-Fold cross-validation (5 katmanlı)
#                            n_jobs=-1,  # Paralel işlem için -1 kullanılır, tüm CPU çekirdeklerini kullanır
#                            verbose=2)  # Detaylı bilgi için

# # Grid Search ile modelin eğitimini yap
# grid_search.fit(X_train, y_train)

# # En iyi parametreleri yazdır
# print(f"En iyi parametreler: {grid_search.best_params_}")

# # En iyi modeli al
# best_rf_model = grid_search.best_estimator_

# # Test verisi üzerinde tahmin yap
# test_predictions = best_rf_model.predict(X_test)

# # Modelin performansını değerlendirme
# test_mae = mean_absolute_error(y_test, test_predictions)
# test_mse = mean_squared_error(y_test, test_predictions)

# print(f"Test MAE: {test_mae:.2f}")
# print(f"Test MSE: {test_mse:.2f}")

# # GridSearchCV sonuçlarını yazdırma
# print("GridSearchCV Sonuçları:")
# print("En iyi skoru veren parametreler:")
# print(grid_search.best_params_)

# # Tüm parametre kombinasyonları için sonuçları yazdır
# cv_results = pd.DataFrame(grid_search.cv_results_)
# print(cv_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])




# 6. Random Forest Modelini Tanımlayın
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, max_features='sqrt', min_samples_leaf=4, min_samples_split=10)
rf_model.fit(X_train, y_train)

# 7. Test Verisinde Tahmin Yapın
test_predictions = rf_model.predict(X_test)
# 8. Tahminleri Değerlendirin
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f"Random Forest Test MAE: {test_mae:.2f}")
print(f"Random Forest Test MSE: {test_mse:.2f}")

# 9. Tahminleri ve Gerçek Değerleri Görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Gerçek Değerler", marker='o')
plt.plot(test_predictions, label="Tahminler", marker='x')
plt.title("Test Verisi: Gerçek ve Tahmin Edilen Değerler (Random Forest)")
plt.xlabel("Veri Noktası")
plt.ylabel("Sıcaklık (°C)")
plt.legend()
plt.savefig('rf_time_series_comparison.png')
plt.show()

# 10. Regresyon ve Hata Dağılımı Grafikleri
# (a) Regresyon Grafiği
plt.figure(figsize=(8, 8))
plt.scatter(y_test, test_predictions, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Doğru Çizgi")
plt.title("Regresyon Grafiği: Gerçek vs Tahmin (Random Forest)")
plt.xlabel("Gerçek Sıcaklık (°C)")
plt.ylabel("Tahmin Edilen Sıcaklık (°C)")
plt.legend()
plt.savefig('rf_regression_plot.png')
plt.show()

# (b) Hata Dağılımı Grafiği
errors = y_test - test_predictions
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title("Hata Dağılımı (Random Forest)")
plt.xlabel("Hata (Gerçek - Tahmin)")
plt.ylabel("Frekans")
plt.savefig('rf_error_distribution.png')
plt.show()

# 11. MAE ve MSE Sonuçlarını Kaydet
results = {
    "Metric": ["MAE", "MSE"],
    "Value": [test_mae, test_mse]
}

results_df = pd.DataFrame(results)
results_df.to_csv("model_performance_metrics_rf.csv", index=False)
print("MAE ve MSE sonuçları 'model_performance_metrics_rf.csv' dosyasına kaydedildi.")

# Tahminleri ve gerçek değerleri kaydet
results_df = pd.DataFrame({
    'True Temperature': y_test,
    'Predicted Temperature': test_predictions
})
results_df.to_csv('true_vs_predicted_temperatures_rf.csv', index=False)