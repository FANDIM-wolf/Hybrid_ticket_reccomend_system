import pandas as pd

# Загрузка данных
file_path = 'combined_prices_for_dates_roundtrip.csv'
df = pd.read_csv(file_path)

# Приводим колонку с изменением цены к числовому типу (нечисловые значения станут NaN)
df['delta_next_days'] = pd.to_numeric(df['delta_next_days'], errors='coerce')

# Отбираем строки, где изменение цены ненулевое и не пропущено
changed_rows = df[df['delta_next_days'].notna() & (df['delta_next_days'] != 0)]

# Количество уникальных рейсов (по ticket_id), у которых было хотя бы одно изменение
flights_with_change = changed_rows['ticket_id'].nunique()

# Для справки: общее количество уникальных рейсов в датасете
total_flights = df['ticket_id'].nunique()

print(f"Всего уникальных рейсов в данных: {total_flights}")
print(f"Рейсов, у которых цена менялась хотя бы один раз: {flights_with_change}")
print(f"Процент рейсов с изменениями: {flights_with_change / total_flights * 100:.2f}%")