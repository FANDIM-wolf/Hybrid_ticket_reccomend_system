import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Загрузка
df = pd.read_csv('train_lstm.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['option', 'date', 'time_encoded']).reset_index(drop=True)

# Так как Random Forest не умеет в последовательности сам,
# мы вручную делаем сдвиг таргета на 1 шаг назад (предсказываем следующую точку)
df['target'] = df.groupby('option')['price_change_percent'].shift(-1)
df = df.dropna(subset=['target']) # Убираем последние точки (их предсказывать не на чем)

feature_cols = ['usd_rate', 'day_of_week', 'days_until_departure', 'time_encoded', 'best_price']
target_col = 'target'

X = df[feature_cols]
y = df[target_col]

# Разбиваем по времени (первые 70% - трейн, остальные - тест) для каждого option
train_idx, test_idx = [], []
for _, group in df.groupby('option'):
    n = len(group)
    train_idx.extend(group.iloc[:int(n*0.8)].index)
    test_idx.extend(group.iloc[int(n*0.8):].index)

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_test, y_test = X.loc[test_idx], y.loc[test_idx]

# Масштабирование (для леса не строго обязательно, но пусть будет)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели (0 усложнений, просто лес)
model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# Оценка
preds_train = model.predict(X_train_scaled)
preds_test = model.predict(X_test_scaled)

print(f"Train R2: {r2_score(y_train, preds_train):.3f}")
print(f"Test  R2: {r2_score(y_test, preds_test):.3f}")
print(f"Test MAE: {mean_absolute_error(y_test, preds_test):.3f}")