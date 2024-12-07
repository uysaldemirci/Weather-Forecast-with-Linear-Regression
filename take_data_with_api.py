import requests
import time
import calendar
import pandas as pd
from datetime import datetime, timedelta

# API bilgileri
API_KEY = "970d5a6ce26cfd9e865c85e580b90116"  
lat = 41.0351  
lon = 28.9833  

# Tarih aralığını belirle
start_date = datetime.strptime('2024-10-28', '%Y-%m-%d')  # Başlangıç tarihi
end_date = datetime.strptime('2024-11-30', '%Y-%m-%d')  # Bitiş tarihi

# Tüm veriler için bir liste
all_data = []

# 7 günlük aralıklarla API çağrısı yap
current_start = start_date
while current_start < end_date:
    # 7 günlük aralık
    current_end = current_start + timedelta(days=7)
    # Bitiş tarihi, genel bitiş tarihini aşmasın
    if current_end > end_date:
        current_end = end_date
    
    # Unix zaman damgalarına dönüştür
    start = calendar.timegm(current_start.timetuple())
    end = calendar.timegm(current_end.timetuple())
    
    # API çağrısı
    url = f"https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start}&end={end}&appid={API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        print(f"Successful API call for {current_start} to {current_end}")
        
        # JSON verisini al ve listeye ekle
        data = response.json()
        all_data.extend(data.get("list", []))  # "list" anahtarındaki tüm verileri ekle
    else:
        print(f"Error: {response.status_code}, {response.text}")
    
    # Bir sonraki aralık için başlangıç tarihini güncelle
    current_start = current_end

# Tüm verileri DataFrame'e dönüştür
df = pd.json_normalize(all_data)

# Zaman damgasını insan tarafından okunabilir hale çevir
df["dt"] = pd.to_datetime(df["dt"], unit="s")

# CSV olarak kaydet
df.to_csv("weather_data_full_year2.csv", index=False)

print("Full year data saved to weather_data_full_year.csv")

