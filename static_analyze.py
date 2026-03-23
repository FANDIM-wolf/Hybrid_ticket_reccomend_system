#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ динамики цен на авиабилеты и построение модели прогнозирования
Версия 2.2: CatBoost + разделение 60% Train / 25% Validation / 15% Test
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Попытка импорта CatBoost
try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost не установлен. Установите: pip install catboost")

# ============================================
# 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
# ============================================

def load_and_prepare_data(filepath='combined_prices_for_dates_oneway.csv'):
    """Загрузка и подготовка данных"""

    # Читаем без имён колонок
    df = pd.read_csv(filepath, header=None)

    print(f"📊 Размер данных: {df.shape[0]} строк × {df.shape[1]} колонок")

    # Названия ключевых колонок по индексам
    df = df.rename(columns={
        0: 'flight_number',
        1: 'departure_airport',
        2: 'arrival_airport',
        3: 'depart_date',
        4: 'airline',
        5: 'route_direction',
        6: 'city_pair',
        7: 'price',
        8: 'tax',
        9: 'distance',
        10: 'duration',
        14: 'observation_ts',
        19: 'origin',
        20: 'destination',
        63: 'departure_day_of_week',
        65: 'observation_day_of_week',
    })

    # Конвертация типов
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['depart_date'] = pd.to_datetime(df['depart_date'], errors='coerce')
    df['observation_ts'] = pd.to_datetime(df['observation_ts'], errors='coerce')

    # Удаление NaN
    df = df.dropna(subset=['price', 'depart_date', 'observation_ts'])

    return df

# ============================================
# 2. ФИЛЬТРАЦИЯ ПО МАРШРУТАМ
# ============================================

def filter_routes(df):
    """Фильтрация по целевым маршрутам"""

    routes_config = {
        'MOW_TAS': {'origin': 'MOW', 'destination': 'TAS', 'name': 'Москва → Ташкент'},
        'MOW_SHA': {'origin': 'MOW', 'destination': 'SHA', 'name': 'Москва → Шанхай'},
        'MOW_NQZ': {'origin': 'MOW', 'destination': 'NQZ', 'name': 'Москва → Нур-Султан'},
        'KZN_MOW': {'origin': 'KZN', 'destination': 'MOW', 'name': 'Казань → Москва'},
        'MOW_KRR': {'origin': 'MOW', 'destination': 'KRR', 'name': 'Москва → Краснодар'},
        'KRR_MOW': {'origin': 'KRR', 'destination': 'MOW', 'name': 'Краснодар → Москва'},
        'MOW_LED': {'origin': 'MOW', 'destination': 'LED', 'name': 'Москва → Санкт-Петербург'},
        'LED_MOW': {'origin': 'LED', 'destination': 'MOW', 'name': 'Санкт-Петербург → Москва'},
        'MOW_TJM': {'origin': 'MOW', 'destination': 'TJM', 'name': 'Москва → Тюмень'},
        'TJM_MOW': {'origin': 'TJM', 'destination': 'MOW', 'name': 'Тюмень → Москва'},
    }

    filtered_data = {}

    for route_key, config in routes_config.items():
        mask = (df['origin'] == config['origin']) & (df['destination'] == config['destination'])
        route_df = df[mask].copy()

        if len(route_df) > 0:
            # Добавляем дни до вылета
            route_df['days_before_departure'] = (
                    route_df['depart_date'] - route_df['observation_ts']
            ).dt.days

            # Добавляем номер наблюдения
            route_df = route_df.sort_values('observation_ts')
            route_df['snapshot_number'] = range(1, len(route_df) + 1)

            filtered_data[route_key] = {
                'data': route_df,
                'config': config
            }

    print(f"\n🔍 Найдено маршрутов: {len(filtered_data)}")
    for route_key, info in filtered_data.items():
        print(f"  • {info['config']['name']}: {len(info['data'])} наблюдений")

    return filtered_data

# ============================================
# 3. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================

def analyze_route_statistics(route_data):
    """Статистический анализ по маршруту"""

    results = {}

    for route_key, info in route_data.items():
        df = info['data']
        config = info['config']

        stats = {
            'route_name': config['name'],
            'total_observations': len(df),
            'unique_flights': df['flight_number'].nunique(),
            'unique_airlines': df['airline'].nunique(),
            'price_min': df['price'].min(),
            'price_max': df['price'].max(),
            'price_mean': df['price'].mean(),
            'price_median': df['price'].median(),
            'price_std': df['price'].std(),
            'price_cv': df['price'].std() / df['price'].mean() * 100,
        }

        results[route_key] = stats

    return results

# ============================================
# 4. ПОДГОТОВКА ПРИЗНАКОВ ДЛЯ МОДЕЛИ
# ============================================

def prepare_features(df):
    """Подготовка признаков для модели"""

    df = df.copy()

    # Дни до вылета (основной признак)
    df['days_before_departure'] = df['days_before_departure'].fillna(0)

    # День недели вылета
    df['departure_day_of_week'] = df['depart_date'].dt.dayofweek.fillna(0)

    # Месяц вылета
    df['departure_month'] = df['depart_date'].dt.month.fillna(0)

    # Выходной ли день вылета
    df['is_weekend_departure'] = df['departure_day_of_week'].isin([5, 6]).astype(int)

    # Номер рейса (как категория)
    df['flight_number_encoded'] = df['flight_number'].astype(str).str[:3].astype(int)

    # Авиакомпания (кодировка)
    df['airline_encoded'] = df['airline'].astype('category').cat.codes

    # Аэропорт вылета (кодировка)
    df['departure_airport_encoded'] = df['departure_airport'].astype('category').cat.codes

    # Час вылета
    df['departure_hour'] = df['depart_date'].dt.hour.fillna(0)

    # Квадрат дней до вылета (нелинейность)
    df['days_squared'] = df['days_before_departure'] ** 2

    # Логарифм дней до вылета
    df['days_log'] = np.log1p(df['days_before_departure'])

    return df

def get_feature_columns():
    """Список колонок признаков"""
    return [
        'days_before_departure',
        'departure_day_of_week',
        'departure_month',
        'is_weekend_departure',
        'flight_number_encoded',
        'airline_encoded',
        'departure_airport_encoded',
        'departure_hour',
        'days_squared',
        'days_log'
    ]

# ============================================
# 5. ОБУЧЕНИЕ МОДЕЛЕЙ С РАЗДЕЛЕНИЕМ 60/25/15
# ============================================

def train_models(route_data, route_key='MOW_TAS', save_dir='models'):
    """Обучение моделей с правильным разделением данных"""

    if route_key not in route_data:
        print(f"❌ Маршрут {route_key} не найден!")
        return None

    df = route_data[route_key]['data'].copy()
    config = route_data[route_key]['config']

    # Подготовка признаков
    df = prepare_features(df)
    df = df.dropna(subset=['price'])

    if len(df) < 50:
        print(f"⚠️ Недостаточно данных для обучения (всего {len(df)} записей)")
        return None

    # Признаки и целевая переменная
    feature_cols = get_feature_columns()

    # ============================================
    # 🔧 ХРОНОЛОГИЧЕСКОЕ РАЗДЕЛЕНИЕ (БЕЗ SHUFFLE!)
    # ============================================

    # 1. Сортируем ПО ВРЕМЕНИ НАБЛЮДЕНИЯ
    df_sorted = df.sort_values('observation_ts').reset_index(drop=True)

    X_full = df_sorted[feature_cols].fillna(0)
    y_full = df_sorted['price']

    # 2. Рассчитываем индексы для 60/25/15
    n = len(X_full)
    train_end = int(n * 0.60)   # 60% Train
    val_end = int(n * 0.85)     # 85% (Train + Val)

    # 3. Разделяем ХРОНОЛОГИЧЕСКИ
    X_train = X_full.iloc[:train_end]
    y_train = y_full.iloc[:train_end]

    X_val = X_full.iloc[train_end:val_end]
    y_val = y_full.iloc[train_end:val_end]

    X_test = X_full.iloc[val_end:]
    y_test = y_full.iloc[val_end:]

    print(f"\n📊 Данные для обучения ({config['name']}):")
    print(f"   Всего записей: {n}")
    print(f"   Train: {len(X_train)} записей ({len(X_train)/n*100:.1f}%)")
    print(f"   Validation: {len(X_val)} записей ({len(X_val)/n*100:.1f}%)")
    print(f"   Test: {len(X_test)} записей ({len(X_test)/n*100:.1f}%)")
    print(f"\n📅 Временной диапазон:")
    print(f"   Train: {df_sorted['observation_ts'].iloc[0]} — {df_sorted['observation_ts'].iloc[train_end-1]}")
    print(f"   Val:   {df_sorted['observation_ts'].iloc[train_end]} — {df_sorted['observation_ts'].iloc[val_end-1]}")
    print(f"   Test:  {df_sorted['observation_ts'].iloc[val_end]} — {df_sorted['observation_ts'].iloc[-1]}")
    print(f"   Признаков: {len(feature_cols)}")

    # ============================================
    # ДАЛЬШЕ ОБУЧЕНИЕ МОДЕЛЕЙ (без изменений)
    # ============================================

    os.makedirs(save_dir, exist_ok=True)
    results = {}

    # 1. Linear Regression
    print("\n🔹 Обучение Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred_val = lr_model.predict(X_val)
    lr_pred_test = lr_model.predict(X_test)

    results['LinearRegression'] = {
        'model': lr_model,
        'val_mae': mean_absolute_error(y_val, lr_pred_val),
        'val_rmse': np.sqrt(mean_squared_error(y_val, lr_pred_val)),
        'val_r2': r2_score(y_val, lr_pred_val),
        'test_mae': mean_absolute_error(y_test, lr_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, lr_pred_test)),
        'test_r2': r2_score(y_test, lr_pred_test),
        'coefficients': dict(zip(feature_cols, lr_model.coef_)),
        'intercept': lr_model.intercept_
    }

    joblib.dump(lr_model, os.path.join(save_dir, f'{route_key}_linear.joblib'))

    # 2. Random Forest
    print("🔹 Обучение Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred_val = rf_model.predict(X_val)
    rf_pred_test = rf_model.predict(X_test)

    results['RandomForest'] = {
        'model': rf_model,
        'val_mae': mean_absolute_error(y_val, rf_pred_val),
        'val_rmse': np.sqrt(mean_squared_error(y_val, rf_pred_val)),
        'val_r2': r2_score(y_val, rf_pred_val),
        'test_mae': mean_absolute_error(y_test, rf_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, rf_pred_test)),
        'test_r2': r2_score(y_test, rf_pred_test),
        'feature_importance': dict(zip(feature_cols, rf_model.feature_importances_))
    }

    joblib.dump(rf_model, os.path.join(save_dir, f'{route_key}_randomforest.joblib'))

    # 3. CatBoost
    if CATBOOST_AVAILABLE:
        print("🔹 Обучение CatBoost...")
        cat_model = CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            loss_function='RMSE',
            verbose=False,
            random_state=42,
            early_stopping_rounds=50,
            use_best_model=True
        )

        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        cat_model.fit(train_pool, eval_set=val_pool, verbose=False)

        cat_pred_val = cat_model.predict(X_val)
        cat_pred_test = cat_model.predict(X_test)

        results['CatBoost'] = {
            'model': cat_model,
            'val_mae': mean_absolute_error(y_val, cat_pred_val),
            'val_rmse': np.sqrt(mean_squared_error(y_val, cat_pred_val)),
            'val_r2': r2_score(y_val, cat_pred_val),
            'test_mae': mean_absolute_error(y_test, cat_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, cat_pred_test)),
            'test_r2': r2_score(y_test, cat_pred_test),
            'feature_importance': dict(zip(feature_cols, cat_model.feature_importances_))
        }

        cat_model.save_model(os.path.join(save_dir, f'{route_key}_catboost.cbm'))
    else:
        results['CatBoost'] = None
        print("⚠️ CatBoost пропущен (не установлен)")

    # Сохранение метаданных
    metadata = {
        'route_key': route_key,
        'route_name': config['name'],
        'feature_columns': feature_cols,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'train_ratio': len(X_train)/n,
        'val_ratio': len(X_val)/n,
        'test_ratio': len(X_test)/n,
        'train_date': datetime.now().isoformat(),
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'model'}
                    for k, v in results.items() if v is not None}
    }
    joblib.dump(metadata, os.path.join(save_dir, f'{route_key}_metadata.joblib'))

    return results, X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

# ============================================
# 6. ВЫВОД ВХОДНЫХ/ВЫХОДНЫХ ДАННЫХ МОДЕЛИ
# ============================================

def print_model_io_data(X_train, X_val, X_test, y_train, y_val, y_test, results, feature_cols, route_name):
    """Вывод входных и выходных данных моделей"""

    print("\n" + "=" * 80)
    print(f"📋 ВХОДНЫЕ И ВЫХОДНЫЕ ДАННЫЕ МОДЕЛИ: {route_name}")
    print("=" * 80)

    # Входные данные
    print("\n📥 ВХОДНЫЕ ДАННЫЕ (Признаки):")
    print("-" * 80)
    print(f"Количество признаков: {len(feature_cols)}")
    print(f"Названия признаков: {feature_cols}")
    print(f"\nРазмерность train: {X_train.shape} (60%)")
    print(f"Размерность validation: {X_val.shape} (25%)")
    print(f"Размерность test: {X_test.shape} (15%)")

    print("\n📊 Статистика признаков (train):")
    stats_df = pd.DataFrame({
        'Признак': feature_cols,
        'Среднее': [X_train[col].mean() for col in feature_cols],
        'Мин': [X_train[col].min() for col in feature_cols],
        'Макс': [X_train[col].max() for col in feature_cols],
        'Стд': [X_train[col].std() for col in feature_cols]
    })
    print(stats_df.to_string(index=False))

    print("\n📄 Пример входных данных (первые 5 записей train):")
    print(X_train.head().to_string())

    # Выходные данные
    print("\n\n📤 ВЫХОДНЫЕ ДАННЫЕ (Целевая переменная - цена):")
    print("-" * 80)
    print(f"Размерность y_train: {y_train.shape}")
    print(f"Размерность y_val: {y_val.shape}")
    print(f"Размерность y_test: {y_test.shape}")
    print(f"\nСтатистика цены (train):")
    print(f"   Мин: {y_train.min():.2f} ₽")
    print(f"   Макс: {y_train.max():.2f} ₽")
    print(f"   Среднее: {y_train.mean():.2f} ₽")
    print(f"   Медиана: {y_train.median():.2f} ₽")
    print(f"   Стд: {y_train.std():.2f} ₽")

    # Результаты моделей
    print("\n\n🎯 РЕЗУЛЬТАТЫ МОДЕЛЕЙ:")
    print("-" * 80)

    summary_data = []
    for model_name, model_data in results.items():
        if model_data is None:
            continue
        summary_data.append({
            'Модель': model_name,
            'Val MAE (₽)': f"{model_data['val_mae']:.2f}",
            'Val RMSE (₽)': f"{model_data['val_rmse']:.2f}",
            'Val R²': f"{model_data['val_r2']:.4f}",
            'Test MAE (₽)': f"{model_data['test_mae']:.2f}",
            'Test RMSE (₽)': f"{model_data['test_rmse']:.2f}",
            'Test R²': f"{model_data['test_r2']:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Важность признаков (для Random Forest и CatBoost)
    print("\n\n📊 ВАЖНОСТЬ ПРИЗНАКОВ:")
    print("-" * 80)

    for model_name in ['RandomForest', 'CatBoost']:
        if results.get(model_name) and 'feature_importance' in results[model_name]:
            importance = results[model_name]['feature_importance']
            importance_df = pd.DataFrame({
                'Признак': list(importance.keys()),
                'Важность (%)': [v * 100 for v in importance.values()]
            }).sort_values('Важность (%)', ascending=False)

            print(f"\n{model_name}:")
            print(importance_df.to_string(index=False))

    # Коэффициенты Linear Regression
    if results.get('LinearRegression'):
        coef = results['LinearRegression']['coefficients']
        coef_df = pd.DataFrame({
            'Признак': list(coef.keys()),
            'Коэффициент': list(coef.values())
        }).sort_values('Коэффициент', ascending=False)

        print(f"\nLinear Regression:")
        print(f"Intercept (свободный член): {results['LinearRegression']['intercept']:.2f}")
        print(coef_df.to_string(index=False))

    # Пример предсказаний
    print("\n\n🔮 ПРИМЕР ПРЕДСКАЗАНИЙ (первые 10 записей test):")
    print("-" * 80)

    predictions_df = pd.DataFrame({
        'Фактическая цена': y_test.head(10).values,
    })

    for model_name, model_data in results.items():
        if model_data is None:
            continue
        pred = model_data['model'].predict(X_test.head(10))
        predictions_df[f'{model_name} прогноз'] = pred

    predictions_df['Ошибка RF'] = predictions_df['Фактическая цена'] - predictions_df.get('RandomForest прогноз', 0)
    if results.get('CatBoost'):
        predictions_df['Ошибка CatBoost'] = predictions_df['Фактическая цена'] - predictions_df.get('CatBoost прогноз', 0)

    print(predictions_df.round(2).to_string(index=False))

    return summary_df

# ============================================
# 7. ГЛАВНЫЙ СКРИПТ
# ============================================

def main():
    print("=" * 80)
    print("АНАЛИЗ ДИНАМИКИ ЦЕН НА АВИАБИЛЕТЫ + CatBoost")
    print("Разделение данных: 60% Train / 25% Validation / 15% Test")
    print("=" * 80)

    # Загрузка данных
    print("\n📂 Загрузка данных...")
    df = load_and_prepare_data('combined_prices_for_dates_oneway.csv')
    print(f"✅ Загружено {len(df)} записей")

    # Фильтрация маршрутов
    print("\n🔍 Фильтрация по маршрутам...")
    route_data = filter_routes(df)

    if not route_data:
        print("\n❌ Маршруты не найдены!")
        return

    # Статистический анализ
    print("\n📊 Статистический анализ...")
    stats = analyze_route_statistics(route_data)

    print("\n" + "=" * 80)
    print("СВОДНАЯ СТАТИСТИКА ПО МАРШРУТАМ")
    print("=" * 80)

    for route_key, s in stats.items():
        print(f"\n✈️ {s['route_name']}")
        print(f"   Наблюдений: {s['total_observations']}")
        print(f"   Цена: {s['price_min']:,.0f} - {s['price_max']:,.0f} ₽ (средняя: {s['price_mean']:,.0f} ₽)")
        print(f"   Волатильность (CV): {s['price_cv']:.2f}%")

    # Обучение моделей для каждого маршрута
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)

    all_models = {}

    for route_key in route_data.keys():
        print(f"\n{'='*60}")
        print(f"🚀 Маршрут: {route_data[route_key]['config']['name']}")
        print(f"{'='*60}")

        models_result = train_models(route_data, route_key, save_dir='models')

        if models_result:
            results, X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = models_result

            # Вывод входных/выходных данных
            print_model_io_data(
                X_train, X_val, X_test, y_train, y_val, y_test,
                results, feature_cols,
                route_data[route_key]['config']['name']
            )

            all_models[route_key] = results

    # Итоговая сводка
    print("\n" + "=" * 80)
    print("📁 СОХРАНЁННЫЕ МОДЕЛИ")
    print("=" * 80)

    if os.path.exists('models'):
        model_files = os.listdir('models')
        for f in sorted(model_files):
            print(f"  📄 models/{f}")

    print("\n" + "=" * 80)
    print("✅ АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 80)

    print("\n💡 Рекомендации:")
    print("   • CatBoost показывает лучшую точность на табличных данных")
    print("   • Для LSTM нужно собирать данные 60-90 дней с частотой каждые 6 часов")
    print("   • Разделение 60/25/15 обеспечивает надёжную валидацию модели")

    return stats, all_models

if __name__ == '__main__':
    stats, all_models = main()