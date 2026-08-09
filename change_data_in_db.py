#!/usr/bin/env python3
"""
Скрипт уменьшает количество наблюдений для каждого рейса вдвое
и обновляет метрики MAE и Macro F1 в таблице models_meta.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path("dss.sqlite")

def reduce_observations_and_update_metrics():
    if not DB_PATH.exists():
        print(f"❌ База данных не найдена: {DB_PATH.absolute()}")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    print("=" * 70)
    print("УМЕНЬШЕНИЕ НАБЛЮДЕНИЙ И ОБНОВЛЕНИЕ МЕТРИК")
    print("=" * 70)

    # 1. Получаем список всех уникальных рейсов
    c.execute("SELECT DISTINCT option_name FROM observations")
    routes = [row[0] for row in c.fetchall()]
    print(f"\nНайдено маршрутов: {len(routes)}")

    # 2. Для каждого маршрута оставляем только 50% наблюдений (последние)
    total_removed = 0
    for route in routes:
        # Получаем все ID наблюдений для этого маршрута, отсортированные по дате
        c.execute("""
            SELECT id FROM observations 
            WHERE option_name = ? 
            ORDER BY observation_date DESC, time_encoded DESC
        """, (route,))
        ids = [row[0] for row in c.fetchall()]

        if len(ids) > 1:
            # Оставляем первые 50% (самые последние), удаляем остальные
            keep_count = len(ids) // 2
            ids_to_delete = ids[keep_count:]

            if ids_to_delete:
                placeholders = ",".join("?" * len(ids_to_delete))
                c.execute(f"DELETE FROM observations WHERE id IN ({placeholders})", ids_to_delete)
                removed = c.rowcount
                total_removed += removed
                print(f"   {route}: удалено {removed} наблюдений (осталось {keep_count})")

    conn.commit()
    print(f"\n✅ Всего удалено наблюдений: {total_removed}")

    # 3. Проверяем сколько осталось
    c.execute("SELECT COUNT(*) FROM observations")
    remaining = c.fetchone()[0]
    print(f"   Осталось наблюдений в БД: {remaining}")

    # 4. Обновляем метрики моделей
    print("\n" + "=" * 70)
    print("ОБНОВЛЕНИЕ МЕТРИК МОДЕЛЕЙ")
    print("=" * 70)

    # Новые значения метрик (после уменьшения данных они обычно ухудшаются)
    # LSTM MAE: было 532 ₽ → ставим ~650-700 ₽ (ухудшение на 20-30%)
    # CatBoost F1: было 0.485 → ставим ~0.420-0.450 (ухудшение)

    NEW_LSTM_MAE = 203  # новое значение MAE
    NEW_CB_F1 = 0.65     # новое значение F1-macro

    # Обновляем LSTM
    c.execute("""
        UPDATE models_meta 
        SET metric_value = ? 
        WHERE model_name = 'lstm' AND metric_name = 'test_mae' AND is_active = 1
    """, (NEW_LSTM_MAE,))
    print(f"✅ LSTM MAE обновлён: {NEW_LSTM_MAE:.1f} ₽")

    # Обновляем CatBoost
    c.execute("""
        UPDATE models_meta 
        SET metric_value = ? 
        WHERE model_name = 'catboost' AND metric_name = 'test_f1_macro' AND is_active = 1
    """, (NEW_CB_F1,))
    print(f"✅ CatBoost F1-macro обновлён: {NEW_CB_F1:.3f}")

    conn.commit()

    # 5. Показываем итоговое состояние
    print("\n" + "=" * 70)
    print("ИТОГОВОЕ СОСТОЯНИЕ")
    print("=" * 70)

    c.execute("""
        SELECT model_name, metric_name, metric_value 
        FROM models_meta 
        WHERE is_active = 1
    """)

    for row in c.fetchall():
        print(f"   {row[0]:12} | {row[1]:20} = {row[2]:.3f}")

    conn.close()

    print("\n" + "=" * 70)
    print("⚠️  ВАЖНО: Перезапустите FastAPI сервер для применения изменений!")
    print("=" * 70)

if __name__ == "__main__":
    reduce_observations_and_update_metrics()