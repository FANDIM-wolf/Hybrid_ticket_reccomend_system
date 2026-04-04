import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. ЗАГРУЗКА И ПРОВЕРКА
# ==========================================
df = pd.read_csv('train_lstm.csv')

# ЗАЩИТНАЯ ПРОВЕРКА: Если файл пустой
if len(df) == 0:
    raise ValueError("ФАТАЛЬНАЯ ОШИБКА: Файл 'train_lstm.csv' пустой! В нем нет данных (только заголовки). Проверьте файл.")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['option', 'date', 'time_encoded']).reset_index(drop=True)

def get_time_delta(time_str):
    return pd.Timedelta(hours=9) if time_str == 'morning' else pd.Timedelta(hours=18)
df['plot_datetime'] = df['date'] + df['time'].apply(get_time_delta)

for lag in [1, 2]:
    df[f'lag_{lag}_change'] = df.groupby('option')['price_change_percent'].shift(lag)
    df[f'lag_{lag}_price'] = df.groupby('option')['best_price'].shift(lag)
df = df.fillna(0)

feature_cols = [
    'usd_rate', 'day_of_week', 'days_until_departure', 'time_encoded', 'best_price',
    'lag_1_change', 'lag_2_change', 'lag_1_price', 'lag_2_price'
]
target_col = 'price_change_percent'

# ==========================================
# 2. МАСШТАБИРОВАНИЕ
# ==========================================
train_indices = []
for option_name, group in df.groupby('option'):
    train_end = int(len(group) * 0.75)
    train_indices.extend(group.iloc[:train_end].index.tolist())

scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(df.loc[train_indices, feature_cols].values)
scaler_y.fit(df.loc[train_indices, target_col].values.reshape(-1, 1))

X_scaled = scaler_X.transform(df[feature_cols].values)
y_scaled = scaler_y.transform(df[target_col].values.reshape(-1, 1)).flatten()
df['X_scaled'] = list(X_scaled)
df['y_scaled'] = y_scaled

SEQ_LENGTH = 2

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

# ==========================================
# 3. ГЛОБАЛЬНОЕ ОБУЧЕНИЕ
# ==========================================
X_train_global, y_train_global = [], []
for option_name, group in df.groupby('option'):
    train_end = int(len(group) * 0.75)
    X_arr = np.stack(group['X_scaled'].values)
    y_arr = group['y_scaled'].values
    X_tr, y_tr = create_sequences(X_arr[:train_end], y_arr[:train_end], SEQ_LENGTH)
    if len(X_tr) > 0: X_train_global.append(X_tr); y_train_global.append(y_tr)

X_train = np.concatenate(X_train_global)
y_train = np.concatenate(y_train_global)

model = Sequential([
    LSTM(32, activation='tanh', return_sequences=True, input_shape=(SEQ_LENGTH, len(feature_cols))),
    LSTM(16, activation='tanh', return_sequences=False),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

print("ОБУЧЕНИЕ МОДЕЛИ НА ВСЕХ РЯДАХ...")
model.fit(X_train, y_train, epochs=200, batch_size=4, verbose=0)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО.\n")

# ==========================================
# 4. ПОЭЛЕМЕНТНАЯ ОЦЕНКА И ГРАФИКИ
# ==========================================
options = df['option'].unique()
plt.figure(figsize=(15, 20))

for i, opt in enumerate(options):
    group = df[df['option'] == opt].copy()
    n = len(group)
    train_end_idx = int(n * 0.75)

    X_arr = np.stack(group['X_scaled'].values)
    y_arr = group['y_scaled'].values

    X_tr, y_tr = create_sequences(X_arr[:train_end_idx], y_arr[:train_end_idx], SEQ_LENGTH)
    X_te, y_te = create_sequences(X_arr[train_end_idx:], y_arr[train_end_idx:], SEQ_LENGTH)

    pred_tr_scaled = model.predict(X_tr, verbose=0)
    pred_te_scaled = model.predict(X_te, verbose=0) if len(X_te) > 0 else np.array([])

    y_tr_orig = scaler_y.inverse_transform(y_tr.reshape(-1, 1)).flatten()
    pred_tr_orig = scaler_y.inverse_transform(pred_tr_scaled).flatten()
    y_te_orig = scaler_y.inverse_transform(y_te.reshape(-1, 1)).flatten() if len(y_te) > 0 else np.array([])
    pred_te_orig = scaler_y.inverse_transform(pred_te_scaled).flatten() if len(pred_te_scaled) > 0 else np.array([])

    r2_tr = r2_score(y_tr_orig, pred_tr_orig)
    mae_tr = mean_absolute_error(y_tr_orig, pred_tr_orig)
    r2_te = r2_score(y_te_orig, pred_te_orig) if len(y_te_orig) > 1 else np.nan
    mae_te = mean_absolute_error(y_te_orig, pred_te_orig) if len(y_te_orig) > 0 else np.nan

    print(f"Ряд: {opt}")
    print(f"  Train | R2: {r2_tr:.3f} | MAE: {mae_tr:.2f}%")
    print(f"  Test  | R2: {r2_te:.3f} (4 точки) | MAE: {mae_te:.2f}%\n")

    ax = plt.subplot(5, 1, i + 1)

    timeline_tr = group['plot_datetime'].values[SEQ_LENGTH : train_end_idx]
    timeline_te = group['plot_datetime'].values[train_end_idx + SEQ_LENGTH :]

    timeline_plot = np.concatenate([timeline_tr, timeline_te])
    fact_plot = np.concatenate([y_tr_orig, y_te_orig])
    pred_plot = np.concatenate([pred_tr_orig, pred_te_orig])

    split_idx = len(timeline_tr)

    ax.plot(timeline_plot[:split_idx], fact_plot[:split_idx], 'o-', label='Факт (Train)', color='blue', markersize=4)
    ax.plot(timeline_plot[:split_idx], pred_plot[:split_idx], '--', label='Предсказание (Train)', color='blue', alpha=0.6)

    if len(y_te_orig) > 0:
        ax.plot(timeline_plot[split_idx-1:], fact_plot[split_idx-1:], 'o-', label='Факт (Test)', color='red', markersize=6)
        ax.plot(timeline_plot[split_idx-1:], pred_plot[split_idx-1:], 's--', label='Предсказание (Test)', color='red', markersize=6)
        ax.axvline(x=timeline_plot[split_idx-1], color='grey', linestyle=':', alpha=0.5)

    ax.set_title(f"{opt}  |  Train R2: {r2_tr:.2f} | Test R2: {r2_te:.2f}", fontsize=12)
    ax.set_ylabel('Изменение цены (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()