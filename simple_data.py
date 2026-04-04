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

# Генерируем новые лаги прямо здесь (без изменения CSV)
for lag in [1, 2]:
    df[f'lag_{lag}_change'] = df.groupby('option')['price_change_percent'].shift(lag)
    df[f'lag_{lag}_price'] = df.groupby('option')['best_price'].shift(lag)

# Заполняем первые строки нулями
df = df.fillna(0)

# Обновленный список фичей (стало 11 штук!)
feature_cols = [
    'usd_rate', 'day_of_week', 'days_until_departure', 'time_encoded', 'best_price',
    'lag_1_change', 'lag_2_change',  # Изменение 1 и 2 шага назад
    'lag_1_price', 'lag_2_price'     # Абсолютная цена 1 и 2 шага назад
]
target_col = 'price_change_percent'

# ==========================================
# 2. ПОДГОТОВКА ДАННЫХ (80/20)
# ==========================================
train_indices = []
for option_name, group in df.groupby('option'):
    n = len(group)
    if n <= 2: continue
    train_end = int(n * 0.8)
    train_indices.extend(group.iloc[:train_end].index.tolist())

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

def prepare_data(seq_length):
    X_train_g, y_train_g, X_test_g, y_test_g = [], [], [], []
    for option_name, group in df.groupby('option'):
        group = group.copy()
        n = len(group)
        if n <= seq_length: continue
        train_end = int(n * 0.9)

        X_arr = np.stack(group['X_scaled'].values)
        y_arr = group['y_scaled'].values

        X_tr, y_tr = create_sequences(X_arr[:train_end], y_arr[:train_end], seq_length)
        X_te, y_te = create_sequences(X_arr[train_end:], y_arr[train_end:], seq_length)

        if len(X_tr) > 0: X_train_g.append(X_tr); y_train_g.append(y_tr)
        if len(X_te) > 0: X_test_g.append(X_te); y_test_g.append(y_te)

    return (
        np.concatenate(X_train_g), np.concatenate(y_train_g),
        np.concatenate(X_test_g), np.concatenate(y_test_g)
    )

SEQ_LENGTH = 2
X_train, y_train, X_test, y_test = prepare_data(SEQ_LENGTH)

print(f"Размерность X_train: {X_train.shape} (Примеров, Шагов, Фичей)")

# ==========================================
# 3. "ЗЛОДЕЙСКАЯ" АРХИТЕКТУРА (Цель - Переобучение)
# ==========================================
model = Sequential([
    # 1-й слой: 32 нейрона, возвращаем последовательность для следующего слоя
    LSTM(32, activation='tanh', return_sequences=True, input_shape=(SEQ_LENGTH, len(feature_cols))),

    # 2-й слой: 16 нейронов, выдаст только финальный вектор
    LSTM(16, activation='tanh', return_sequences=False),

    # Полносвязный слой для смешивания признаков
    Dense(8, activation='relu'),

    # Выход
    Dense(1, activation='linear')
])

# Увеличиваем скорость обучения (чтобы быстрее схватывал паттерны)
optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

model.summary()

# ==========================================
# 4. ОБУЧЕНИЕ БЕЗ ТОРМОЗОВ
# ==========================================
# EarlyStopping с огромным patience - позволим ей крутиться до победного
early_stop = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=300,           # Много эпох
    batch_size=2,         # Маленький батч (сильнее обновляет веса, лучше переобучает)
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# 5. ОЦЕНКА
# ==========================================
print("\n" + "="*60)
print("------- ДИАГНОСТИКА ПЕРЕОБУЧЕНИЯ -------")
print("="*60)

for name, X_set, y_set in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
    if X_set.shape[0] > 0:
        loss, mae = model.evaluate(X_set, y_set, verbose=0)
        preds = model.predict(X_set, verbose=0)
        r2 = r2_score(y_set, preds)
        print(f"{name:5} | MSE: {loss:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

# Диагностика
train_r2 = r2_score(y_train, model.predict(X_train, verbose=0))
if train_r2 > 0.8:
    print("\n[УСПЕХ] Модель смогла переобучить Train! Архитектура и фичи работают.")
    print("        Теперь можно возвращать Dropout и уменьшать нейроны.")
elif train_r2 > 0.3:
    print("\n[ЧАСТИЧНО] Модель нашла часть закономерностей, но до идеала далеко.")
else:
    print("\n[ПРОВАЛ] Модель не может выучить даже Train. В данных чистый шум.")
print("="*60)