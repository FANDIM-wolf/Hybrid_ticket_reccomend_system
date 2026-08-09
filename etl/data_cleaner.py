"""
Модуль очистки данных от аномалий и несоответствий маршрутов.
Используется в ETL пайплайне перед построением признаков.
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Справочник известных соответствий: Номер рейса -> Ожидаемое направление
# Этот словарь можно расширять по мере выявления новых ошибок API.
# Направления должны совпадать с форматом в поле 'direction' (например, MOW-KRR)
KNOWN_ROUTES = {
    "DP155": "MOW-KRR",
    "DP157": "MOW-KRR",
    "UT573": "MOW-MRV", # UT573 летит в Минводы, а не в Краснодар
    "DP6911": "MOW-MRV",
    "DP6913": "MOW-MRV",
    # Можно добавить другие известные ошибки по мере их обнаружения
}

def clean_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет строки, где номер рейса (flight_number) не соответствует
    заявленному направлению (direction) согласно справочнику KNOWN_ROUTES.

    Args:
        df: DataFrame с сырыми данными из snapshot-файлов.

    Returns:
        Очищенный DataFrame.
    """
    initial_len = len(df)

    if initial_len == 0 or 'flight_number' not in df.columns or 'direction' not in df.columns:
        return df

    # Создаем маску для валидных строк (изначально все True)
    mask = pd.Series([True] * len(df), index=df.index)

    # Проверяем только те рейсы, которые есть в нашем словаре ошибок
    for flight_num, expected_dir in KNOWN_ROUTES.items():
        # Ищем строки с этим номером рейса
        # Приводим flight_number к строке для безопасного сравнения
        flight_mask = df['flight_number'].astype(str) == str(flight_num)

        if flight_mask.any():
            # Среди найденных рейсов проверяем направление
            # Если направление НЕ совпадает с ожидаемым, помечаем как False
            invalid_mask = flight_mask & (df['direction'] != expected_dir)

            if invalid_mask.any():
                count_invalid = invalid_mask.sum()
                logger.warning(
                    f"⚠️ Data Clean: Found {count_invalid} inconsistent rows for flight {flight_num}. "
                    f"Expected direction: {expected_dir}, but found others. Dropping them."
                )
                # Исключаем эти строки из общей маски
                mask &= ~invalid_mask

    cleaned_df = df[mask].copy()
    dropped_count = initial_len - len(cleaned_df)

    if dropped_count > 0:
        logger.info(f"🧹 Data cleaning complete: Dropped {dropped_count} inconsistent rows.")
    else:
        logger.debug("🧹 Data cleaning complete: No inconsistencies found based on known routes.")

    return cleaned_df

def remove_price_outliers(df: pd.DataFrame, threshold_std: float = 3.0) -> pd.DataFrame:
    """
    Удаляет строки, где цена отличается от средней по опции более чем на N стандартных отклонений.
    Группировка по option_name (а не direction) — чтобы не вышибать целые опции,
    у которых ценовой диапазон существенно отличается от других опций того же маршрута.
    """
    initial_len = len(df)

    group_col = "option_name" if "option_name" in df.columns else "direction"

    if initial_len == 0 or 'best_price' not in df.columns or group_col not in df.columns:
        return df

    df_clean = df.copy()
    df_clean['is_outlier'] = False

    for group_val in df_clean[group_col].unique():
        mask_group = df_clean[group_col] == group_val
        prices = df_clean.loc[mask_group, 'best_price']

        if len(prices) < 5:
            continue

        mean_price = prices.mean()
        std_price = prices.std()

        if std_price == 0 or pd.isna(std_price):
            continue

        lower_bound = mean_price - (threshold_std * std_price)
        upper_bound = mean_price + (threshold_std * std_price)

        outliers_mask = (prices < lower_bound) | (prices > upper_bound)
        df_clean.loc[mask_group, 'is_outlier'] = outliers_mask.values

    final_df = df_clean[~df_clean['is_outlier']].drop(columns=['is_outlier'])

    dropped = initial_len - len(final_df)
    if dropped > 0:
        logger.info(f"🧹 Outlier cleaning: Dropped {dropped} rows with anomalous prices (grouped by {group_col}).")

    return final_df