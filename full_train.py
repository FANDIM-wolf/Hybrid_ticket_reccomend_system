import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. ЗАГРУЗКА И "БОМБАРДИРОВКА" ФИЧАМИ
# ==========================================
df = pd.read_csv('train_lstm.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['option', 'date', 'time_encoded']).reset_index(drop=True)

# Генерируем новые лаги
for lag in [1, 2]:
    df[f'lag_{lag}_change'] = df.groupby('option')['price_change_percent'].shift(lag)
    df[f'lag_{lag}_price'] = df.groupby('option')['best_price'].shift(lag)

df = df.fillna(0)

feature_cols = [
    'usd_rate', 'day_of_week', 'days_until_departure', 'time_encoded', 'best_price',
    'lag_1_change', 'lag_2_change',
    'lag_1_price', 'lag_2_price'
]
target_col = 'price_change_percent'

# ==========================================
# 2. ПОДГОТОВКА ДАННЫХ (70 / 20 / 10)
# ==========================================
# ИЗМЕНЕНО: Собираем индексы строго для Train (первые 70%)
train_indices = []
for option_name, group in df.groupby('option'):
    n = len(group)
    if n <= 2: continue
    train_end = int(n * 0.7) # 70% для скейлера
    train_indices.extend(group.iloc[:train_end].index.tolist())

# Обучаем скейлеры ТОЛЬКО на 70% Train
scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(df.loc[train_indices, feature_cols].values)
scaler_y.fit(df.loc[train_indices, target_col].values.reshape(-1, 1))

X_scaled = scaler_X.transform(df[feature_cols].values)
y_scaled = scaler_y.transform(df[target_col].values.reshape(-1, 1)).flatten()
df['X_scaled'] = list(X_scaled)
df['y_scaled'] = y_scaled

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

# ИЗМЕНЕНО: Функция теперь режет на 3 части
def prepare_data(seq_length):
    X_train_g, y_train_g = [], []
    X_val_g, y_val_g = [], []
    X_test_g, y_test_g = [], []

    for option_name, group in df.groupby('option'):
        group = group.copy()
        n = len(group)
        if n <= seq_length: continue

        # Выставляем границы разбиения
        train_end = int(n * 0.7) # 70%
        val_end = int(n * 0.9)   # 70% + 20% = 90%

        X_arr = np.stack(group['X_scaled'].values)
        y_arr = group['y_scaled'].values

        # Режем данные
        X_tr, y_tr = create_sequences(X_arr[:train_end], y_arr[:train_end], seq_length)
        X_v, y_v = create_sequences(X_arr[train_end:val_end], y_arr[train_end:val_end], seq_length)
        X_te, y_te = create_sequences(X_arr[val_end:], y_arr[val_end:], seq_length)

        if len(X_tr) > 0: X_train_g.append(X_tr); y_train_g.append(y_tr)
        if len(X_v) > 0: X_val_g.append(X_v); y_val_g.append(y_v)
        if len(X_te) > 0: X_test_g.append(X_te); y_test_g.append(y_te)

    # Безопасная сборка (на случай если test окажется пустым для каких-то мелких рядов)
    empty_X = np.empty((0, seq_length, len(feature_cols)))
    empty_y = np.empty((0,))

    return (
        np.concatenate(X_train_g) if X_train_g else empty_X,
        np.concatenate(y_train_g) if y_train_g else empty_y,
        np.concatenate(X_val_g) if X_val_g else empty_X,
        np.concatenate(y_val_g) if y_val_g else empty_y,
        np.concatenate(X_test_g) if X_test_g else empty_X,
        np.concatenate(y_test_g) if y_test_g else empty_y
    )

SEQ_LENGTH = 2
X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(SEQ_LENGTH)

print(f"Train shape: {X_train.shape}")
print(f"Val shape:   {X_val.shape}")
print(f"Test shape:  {X_test.shape}\n")

# ==========================================
# 3. "ЗЛОДЕЙСКАЯ" АРХИТЕКТУРА
# ==========================================
model = Sequential([
    LSTM(32, activation='tanh', return_sequences=True, input_shape=(SEQ_LENGTH, len(feature_cols))),
    LSTM(16, activation='tanh', return_sequences=False),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

model.summary()

# ==========================================
# 4. ОБУЧЕНИЕ (EarlyStopping смотрит на Val)
# ==========================================
# ИЗМЕНЕНО: monitor='val_loss' теперь следит за валидацией, а не за тестом!
early_stop = EarlyStopping(monitor='val_loss', patience=170, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val), # <--- Подключили Val
    epochs=400,
    batch_size=2,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# 5. ФИНАЛЬНАЯ ОЦЕНКА (Train, Val, Test)
# ==========================================
print("\n" + "="*60)
print("------- ДИАГНОСТИКА ПЕРЕОБУЧЕНИЯ -------")
print("="*60)

for name, X_set, y_set in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    if X_set.shape[0] > 0:
        loss, mae = model.evaluate(X_set, y_set, verbose=0)
        preds = model.predict(X_set, verbose=0)
        r2 = r2_score(y_set, preds)
        print(f"{name:5} | MSE: {loss:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    else:
        print(f"{name:5} | Выборка пуста (мало точек в ряде)")

# Диагностика
train_r2 = r2_score(y_train, model.predict(X_train, verbose=0))
val_r2 = r2_score(y_val, model.predict(X_val, verbose=0)) if X_val.shape[0] > 0 else 0

print("-" * 60)
if train_r2 > 0.8:
    print("[УСПЕХ] Модель переобучила Train!")
    if val_r2 < 0:
        print("        Но провалила Val. Классический Overfitting (есть паттерн, но нет обобщения).")
    else:
        print("        И показала хороший результат на Val!")
else:
    print("[ПРОВАЛ] Модель не может выучить даже Train.")
print("="*60)