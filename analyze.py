import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. ЗАГРУЗКА И БАЗОВАЯ ПОДГОТОВКА
# ==========================================
try:
    df = pd.read_csv('train_lstm.csv')
except FileNotFoundError:
    print("[ОШИБКА] Файл 'train_lstm.csv' не найден. Проверьте путь.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['option', 'date', 'time_encoded']).reset_index(drop=True)

# Гиперпараметр из вашей модели (влияет на то, сколько данных "съестся" на X)
SEQ_LENGTH = 2

print("=" * 80)
print("                 АНАЛИЗ ВРЕМЕННЫХ РЯДОВ ДАТАСЕТА")
print("=" * 80)

# ==========================================
# 2. ГЛОБАЛЬНАЯ СТАТИСТИКА
# ==========================================
total_rows = len(df)
total_options = df['option'].nunique()

print(f"\nОбщее количество записей (утро/вечер): {total_rows}")
print(f"Количество уникальных временных рядов (опций): {total_options}")
print(f"Среднее количество точек на один ряд: {total_rows / total_options:.1f}")

# ==========================================
# 3. ДЕТАЛЬНАЯ РАЗБИВКА ПО КАЖДОМУ РЯДУ
# ==========================================
analysis_data = []
skipped_options = 0

for option_name, group in df.groupby('option'):
    n = len(group)

    # Считаем индексы разбиения (как в вашем ML коде)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)

    # Сырые точки
    train_points = train_end
    val_points = val_end - train_end
    test_points = n - val_end

    # ВАЖНО: Сколько СЕКВЕНЦИЙ (X,y пар) мы реально получим для LSTM?
    # Если SEQ_LENGTH=2, то из 5 точек мы получим 3 пары: (0,1)->2, (1,2)->3, (2,3)->4
    lstm_train = max(0, train_points - SEQ_LENGTH)
    lstm_val = max(0, val_points - SEQ_LENGTH)
    lstm_test = max(0, test_points - SEQ_LENGTH)

    is_usable = n > SEQ_LENGTH

    if not is_usable:
        skipped_options += 1

    analysis_data.append({
        'Ряд (Option)': option_name,
        'Всего точек': n,
        'Train (сыр)': train_points,
        'Val (сыр)': val_points,
        'Test (сыр)': test_points,
        'LSTM Train (X,y)': lstm_train,
        'LSTM Val (X,y)': lstm_val,
        'LSTM Test (X,y)': lstm_test,
        'Статус': 'ГОДЕН' if is_usable else 'СЛИШКОМ КОРОТКИЙ'
    })

stats_df = pd.DataFrame(analysis_data)

# ==========================================
# 4. ВЫВОД РЕЗУЛЬТАТОВ
# ==========================================
# Выводим таблицу по столбцам
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print("\nДетальная таблица по каждому ряду:")
print("-" * 80)
print(stats_df.to_string(index=False))

# ==========================================
# 5. ИТОГОВАЯ СВОДКА ДЛЯ МОДЕЛИ
# ==========================================
print("\n" + "=" * 80)
print("                      СВОДКА ДЛЯ ОБРАБОТКИ LSTM")
print("=" * 80)

total_lstm_train = stats_df['LSTM Train (X,y)'].sum()
total_lstm_val = stats_df['LSTM Val (X,y)'].sum()
total_lstm_test = stats_df['LSTM Test (X,y)'].sum()

print(f"\nРядов отброшено (длина <= {SEQ_LENGTH}): {skipped_options} шт.")
print(f"Рядов принято в обработку: {total_options - skipped_options} шт.")
print("\nОжидаемые финальные размеры тензоров (после конкатенации):")
print(f"  -> X_train.shape: ({total_lstm_train}, {SEQ_LENGTH}, 5)")
print(f"  -> y_train.shape: ({total_lstm_train},)")
print(f"  -> X_val.shape:   ({total_lstm_val}, {SEQ_LENGTH}, 5)")
print(f"  -> X_test.shape:  ({total_lstm_test}, {SEQ_LENGTH}, 5)")

# Диагностика проблем
print("\n" + "-" * 80)
print("ДИАГНОСТИКА ВОЗМОЖНЫХ ПРОБЛЕМ:")
if total_lstm_train < 50:
    print("⚠️  КРИТИЧЕСКИ МАЛО ТРЕНИРОВОЧНЫХ ДАННЫХ ДЛЯ LSTM (< 50 примеров).")
    print("   Рекомендация: Используйте простые модели (Linear Regression, Random Forest).")

if total_lstm_test == 0:
    print("⚠️  ТЕСТОВАЯ ВЫБОРКА ПУСТА! Модель нельзя адекватно оценить.")
    print("   Рекомендация: Изменить пропорции разбиения (например, 60/20/20) или уменьшить SEQ_LENGTH.")
elif total_lstm_test < 5:
    print("⚠️  Тестовая выборка очень маленькая (< 5 примеров). Метрика R2 будет нестабильной.")

if stats_df['Всего точек'].std() > 3:
    print("⚠️  БОЛЬШОЙ РАЗБРОС В ДЛИНЕ РЯДОВ. Некоторые билеты имеют много истории, другие мало.")
    print("   Это может привести к дисбалансу при обучении.")

print("=" * 80)