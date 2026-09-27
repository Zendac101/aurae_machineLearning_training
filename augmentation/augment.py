import pandas as pd
import numpy as np

# 1. Load data and parse dates
df = pd.read_csv('cleaned_pollutant_data.csv', low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['site_id', 'date']).reset_index(drop=True)

# 2. Impute Missing Values (Linear interpolation for continuous sequences)
pollutant_cols = ['aqi', 'so2', 'co', 'o3',
                  'pm2.5', 'pm10', 'nox', 'aqi_pm25', 'aqi_pm10']
for col in pollutant_cols:
    df[col] = df.groupby('site_id')[col].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both'))

# 3. Time-Based Feature Engineering
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# 4. Lag Features (t-1 historical data)
df['aqi_lag_1'] = df.groupby('site_id')['aqi'].shift(1)
df['pm2.5_lag_1'] = df.groupby('site_id')['pm2.5'].shift(1)
df['pm10_lag_1'] = df.groupby('site_id')['pm10'].shift(1)
df.bfill(inplace=True)  # Fill the initial NaN created by the shift

# 5. Data Augmentation via Jittering (2% Gaussian Noise)
df_synthetic = df.copy()
noise_level = 0.02

for col in pollutant_cols:
    std_dev = df_synthetic[col].std()
    noise = np.random.normal(0, std_dev * noise_level, df_synthetic.shape[0])
    df_synthetic[col] = df_synthetic[col] + noise
    df_synthetic[col] = df_synthetic[col].clip(
        lower=0)  # Prevent negative pollutant values

# 6. Flag and Combine
df['is_synthetic'] = 0
df_synthetic['is_synthetic'] = 1
df_augmented = pd.concat([df, df_synthetic], ignore_index=True)

# Sort chronologically alongside synthetic pairs and export
df_augmented = df_augmented.sort_values(
    by=['site_id', 'date', 'is_synthetic']).reset_index(drop=True)
df_augmented.to_csv("augmented_pollutant_data.csv", index=False)
