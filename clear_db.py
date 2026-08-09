"""
Скрипт для быстрой очистки таблицы наблюдений.
Используйте с осторожностью!
"""
from app.db import get_conn

def clear_observations():
    with get_conn() as c:
        # 1. Считаем сколько было строк
        count_before = c.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

        # 2. Удаляем все строки
        c.execute("DELETE FROM observations")
        c.commit()

        print(f"🗑️ Удалено {count_before} строк из таблицы observations.")
        print("✅ База данных очищена.")

if __name__ == "__main__":
    confirm = input("Вы уверены, что хотите удалить ВСЕ данные? (yes/no): ")
    if confirm.lower() == "yes":
        clear_observations()
    else:
        print("Отмена операции.")