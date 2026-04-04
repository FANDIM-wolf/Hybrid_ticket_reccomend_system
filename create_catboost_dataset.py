import pandas as pd
import numpy as np

# ==========================================
# 1. ЗАГРУЗКА И БАЗОВАЯ ПОДГОТОВКА
# ==========================================
df = pd.read_csv('train_lstm.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['option', 'date', 'time_encoded']).reset_index(drop=True)

# ==========================================
# 2. СОЗДАНИЕ КОЛОНКИ jump_type
# ==========================================
# Берем модуль изменения, так как нас интересует амплитуда, а не знак
abs_change = df['price_change_percent'].abs()

# Условия согласно вашему ТЗ
conditions = [
    abs_change < 0.01,                         # 0 - цена не изменилась
    (abs_change >= 0.01) & (abs_change < 2.0), # 1 - от 0.01% до 2%
    abs_change >= 2.0                           # 2 - от 2% и выше
]
choices = [0, 1, 2]

# np.select позволяет векторизованно (быстро) применить условия ко всему датасету
df['jump_type'] = np.select(conditions, choices, default=0)

# ==========================================
# 3. ПОДГОТОВКА ФИЧЕЙ ДЛЯ CATBOOST (ТАБЛИЧНЫЙ ВИД)
# ==========================================
# CatBoost не умеет в 3D массивы. Делаем сдвиги (лаги) как обычные колонки
for lag in [1, 2]:
    df[f'lag_{lag}_change'] = df.groupby('option')['price_change_percent'].shift(lag)
    df[f'lag_{lag}_price'] = df.groupby('option')['best_price'].shift(lag)

    # БОНУС: Лаг самой классификации! Очень сильная фича для деревьев
    df[f'lag_{lag}_jump_type'] = df.groupby('option')['jump_type'].shift(lag)

# ==========================================
# 4. ФОРМИРОВАНИЕ ТА РГЕТА (ЦЕЛЕВОЙ ПЕРЕМЕННОЙ)
# ==========================================
# Сдвигаем jump_type на -1 (вверх).
# Теперь в строке "Утро 23 марта" таргетом будет тип скачка "Вечер 23 марта".
df['target_jump_type'] = df.groupby('option')['jump_type'].shift(-1)

# Удаляем последние строки (вечер 3 апреля), так как для них нет будущего таргета
df = df.dropna(subset=['target_jump_type'])

# Заполняем NaN в первых строках (утро 23 марта) для лагов
# Для числовых ставим 0, для категориальных ставим -1 (CatBoost поймет)
df = df.fillna({
    'lag_1_change': 0, 'lag_2_change': 0,
    'lag_1_price': 0, 'lag_2_price': 0,
    'lag_1_jump_type': -1, 'lag_2_jump_type': -1
})

# ==========================================
# 5. ФИНАЛЬНАЯ СБОРКА ДАТАСЕТА
# ==========================================
# Оставляем только то, что пойдет в модель
catboost_features = [
    # Категориальные фичи (в CatBoost их нужно будет указать отдельно)
    'option',
    'time',
    'day_of_week',
    'lag_1_jump_type',
    'lag_2_jump_type',

    # Числовые фичи
    'usd_rate',
    'days_until_departure',
    'time_encoded',
    'best_price',
    'lag_1_change',
    'lag_2_change',
    'lag_1_price',
    'lag_2_price',

    # Текущее состояние (можно использовать как фичу)
    'jump_type',

    # ТА РГЕТ
    'target_jump_type'
]

df_cat = df[catboost_features].copy()

# Сохраняем в новый файл
df_cat.to_csv('train_catboost.csv', index=False)

# ==========================================
# 6. ОТЧЕТ
# ==========================================
print("Датасет успешно сохранен как 'train_catboost.csv'")
print(f"Размерность: {df_cat.shape[0]} строк, {df_cat.shape[1]} колонок\n")

print("Распределение ТА РГЕТА (target_jump_type):")
print(df_cat['target_jump_type'].value_counts(normalize=True).round(3) * 100, "%")
print("-" * 40)

print("\nПример данных (первые 5 строк):")
# Для красоты вывода округлим числа
display_cols = ['option', 'time', 'best_price', 'price_change_percent', 'jump_type', 'target_jump_type']
print(df[display_cols].head().to_string(index=False))