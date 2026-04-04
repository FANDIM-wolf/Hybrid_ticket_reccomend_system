import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. ЗАГРУЗКА И БАЗОВАЯ ПОДГОТОВКА
# ==========================================
df = pd.read_csv('train_lstm.csv')
df['date'] = pd.to_datetime(df['date'])
# Строго сортируем по времени внутри каждого билета!
df = df.sort_values(by=['option', 'date', 'time_encoded']).reset_index(drop=True)

# ==========================================
# 2. БЕЗОПАСНАЯ ГЕНЕРАЦИЯ ФИЧЕЙ (Только назад во времени!)
# ==========================================

# --- НОВАЯ ФИЧА: Изменение за 2 предыдущих дня ---
# 2 дня = 4 записи (утро/вечер). Сдвигаем цену на 4 шага назад.
df['price_2_days_ago'] = df.groupby('option')['best_price'].shift(4)
# Считаем процент изменения. Там, где 2 дня назад еще не было данных (первые 4 строки), будет NaN
df['change_2_days_pct'] = ((df['best_price'] - df['price_2_days_ago']) / df['price_2_days_ago']) * 100
df = df.drop(columns=['price_2_days_ago']) # Удаляем промежуточную цену, оставляем только процент

# --- Лаги на 1 и 2 шага (полдня и день назад) ---
for lag in [1, 2]:
    df[f'lag_{lag}_change'] = df.groupby('option')['price_change_percent'].shift(lag)
    df[f'lag_{lag}_price'] = df.groupby('option')['best_price'].shift(lag)
    df[f'lag_{lag}_jump_type'] = df.groupby('option')['jump_type'].shift(lag) if 'jump_type' in df.columns else np.nan

# ==========================================
# 3. ФОРМИРОВАНИЕ ТА РГЕТА (Классификация)
# ==========================================
abs_change = df['price_change_percent'].abs()
conditions = [
    abs_change < 0.01,
    (abs_change >= 0.01) & (abs_change < 2.0),
    abs_change >= 2.0
]
choices = [0, 1, 2]
df['current_jump_type'] = np.select(conditions, choices, default=0)

# ТА РГЕТ: Тип скачка на СЛЕДУЮЩЕМ шаге (shift -1)
df['target_jump_type'] = df.groupby('option')['current_jump_type'].shift(-1)

# Удаляем последние строки каждого ряда (нет будущего таргета)
df = df.dropna(subset=['target_jump_type'])

# Заполняем NaN в начале рядов (где не хватило истории для лагов)
# Числа заполняем 0, новые категориальные лаги заполняем -1
fill_dict = {col: 0 for col in df.select_dtypes(include=[np.number]).columns}
fill_dict.update({'lag_1_jump_type': -1, 'lag_2_jump_type': -1})
df = df.fillna(fill_dict)

# Приводим таргет и типы скачков к целому числу
df['target_jump_type'] = df['target_jump_type'].astype(int)
df['current_jump_type'] = df['current_jump_type'].astype(int)
df['lag_1_jump_type'] = df['lag_1_jump_type'].astype(int)
df['lag_2_jump_type'] = df['lag_2_jump_type'].astype(int)

# ==========================================
# 4. РАЗБИЕНИЕ НА TRAIN И TEST (80% / 20% по времени)
# ==========================================
train_df = pd.DataFrame()
test_df = pd.DataFrame()

# Разбиваем хронологически внутри КАЖДОГО билета отдельно
for option_name, group in df.groupby('option'):
    split_idx = int(len(group) * 0.8)
    train_df = pd.concat([train_df, group.iloc[:split_idx]])
    test_df = pd.concat([test_df, group.iloc[split_idx:]])

print(f"Размер Train: {len(train_df)} строк")
print(f"Размер Test:  {len(test_df)} строк\n")

# ==========================================
# 5. ПОДГОТОВКА ДАННЫХ ДЛЯ CATBOOST
# ==========================================
# Список фичей
features = [
    'usd_rate', 'day_of_week', 'days_until_departure', 'time_encoded', 'best_price',
    'change_2_days_pct', # НОВАЯ ФИЧА
    'lag_1_change', 'lag_2_change',
    'lag_1_price', 'lag_2_price',
    'current_jump_type',
    'lag_1_jump_type', 'lag_2_jump_type'
]

# Категориальные фичи (CatBoost обрабатывает их specially)
cat_features = ['day_of_week', 'current_jump_type', 'lag_1_jump_type', 'lag_2_jump_type']

X_train = train_df[features]
y_train = train_df['target_jump_type']

X_test = test_df[features]
y_test = test_df['target_jump_type']

# Создаем Pool (внутренний формат CatBoost)
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)

# ==========================================
# 6. ОБУЧЕНИЕ CATBOOST
# ==========================================
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=4, # Небольшая глубина (4), чтобы не переобучить на 90 строках
    loss_function='MultiClass',
    auto_class_weights='Balanced', # ВАЖНО: Балансирует классы, так как скачки >2% редкие
    verbose=50 # Печатать лог каждые 50 итераций
)

print("Начало обучения CatBoost...")
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=30)
print("Обучение завершено.\n")

# ==========================================
# 7. ОЦЕНКА КАЧЕСТВА
# ==========================================
y_pred = model.predict(test_pool)
y_pred_flat = y_pred.flatten()

print("="*60)
print("КЛАССИФИКАЦИОННЫЙ ОТЧЕТ НА TEST (20% данных)")
print("="*60)
# target_names=['0: Нет скачка', '1: Мелкий (0-2%)', '2: Большой (>2%)']
print(classification_report(y_test, y_pred_flat, digits=3))

print("="*60)
print("МАТРИЦА ОШИБОК (Confusion Matrix)")
print("="*60)
# Строки - реальный класс, столбцы - предсказанный
cm = confusion_matrix(y_test, y_pred_flat)
print(f"Реальный 0 (нет скачка)   -> Предсказания: {cm[0]}")
print(f"Реальный 1 (мелкий скачок)-> Предсказания: {cm[1]}")
print(f"Реальный 2 (большой скач)-> Предсказания: {cm[2]}")