import pandas as pd
import numpy as np
# Saatlik verilerin olduğu CSV dosyasını yükle
df = pd.read_csv("weather_data_full_year2.csv")

# Zaman damgasını datetime formatına çevir
df["dt"] = pd.to_datetime(df["dt"])

# Tarih sütununu oluştur (sadece gün, ay, yıl)
df["date"] = df["dt"].dt.date

# Günlük ortalamaları hesapla
numeric_cols = df.select_dtypes(include=['number']).columns
daily_data = df.groupby("date")[numeric_cols].mean()

# Eğer datetime index yapmak istemiyorsanız, index'i sıfırlayabilirsiniz
daily_data = daily_data.reset_index()

# Günlük verileri yeni bir CSV dosyasına kaydet
daily_data.to_csv("weather_data_daily2.csv", index=False)

print("Hourly data aggregated to daily data and saved as weather_data_daily.csv")

df = pd.read_csv("weather_data_daily2.csv")

df.drop(columns=['rain.1h', 'wind.gust', 'snow.1h'], inplace = True)

# Tarih bilgisini datetime formatına çevir
df['date'] = pd.to_datetime(df['date'])

# Ay bilgisini çıkar
df['month'] = df['date'].dt.month

# Sinüs ve Kosinüs dönüşümünü yap
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print(df[['date', 'month', 'month_sin', 'month_cos']])

# Mevsimi sayısal olarak ekleyin (1 = Kış, 2 = İlkbahar, 3 = Yaz, 4 = Sonbahar)
df['season'] = df['month'].apply(lambda x: 1 if x in [12, 1, 2] else (2 if x in [3, 4, 5] else (3 if x in [6, 7, 8] else 4)))

# Sinüs ve kosinüs dönüşümü ekleyin
df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4)  # 4 mevsim
df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4)


df.info()

df.to_csv("cleaned_weather_data_for_prediction2.csv", index=False)

print("Columns with missing values are removed")

