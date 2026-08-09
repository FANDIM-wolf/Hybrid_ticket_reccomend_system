#!/usr/bin/env python3
"""Диагностика базы данных - показывает все таблицы и их структуру."""
import sqlite3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Поиск базы данных
possible_db_paths = [

    Path("dss.sqlite"),

]

db_path = None
for p in possible_db_paths:
    if p.exists():
        db_path = p
        break

if db_path is None:
    # Ищем все .db и .sqlite файлы
    candidates = list(Path(".").rglob("*.db")) + list(Path(".").rglob("*.sqlite"))
    if candidates:
        db_path = candidates[0]
        logger.info(f"Найдена БД: {db_path}")
    else:
        logger.error("❌ База данных не найдена!")
        logger.error("Искал в:")
        for p in possible_db_paths:
            logger.error(f"  - {p.absolute()}")
        exit(1)
else:
    logger.info(f"✅ Используемая БД: {db_path.absolute()}")

conn = sqlite3.connect(db_path)
c = conn.cursor()

print("\n" + "="*70)
print("ТАБЛИЦЫ В БАЗЕ ДАННЫХ")
print("="*70)

c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
tables = c.fetchall()

if not tables:
    print("❌ В базе данных нет таблиц!")
else:
    print(f"Найдено таблиц: {len(tables)}")
    for (table_name,) in tables:
        print(f"\n📋 Таблица: {table_name}")

        # Показываем структуру
        c.execute(f"PRAGMA table_info({table_name});")
        columns = c.fetchall()
        print(f"   Колонки ({len(columns)}):")
        for col in columns:
            cid, name, type_, notnull, default, pk = col
            pk_marker = "🔑 " if pk else ""
            print(f"      {pk_marker}{name:30s} {type_:15s} {'NOT NULL' if notnull else ''}")

        # Показываем количество записей
        try:
            c.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = c.fetchone()[0]
            print(f"   Записей: {count}")
        except Exception as e:
            print(f"   ⚠️ Не удалось получить количество: {e}")

# Проверяем, есть ли таблица models_meta или похожие
print("\n" + "="*70)
print("ПОИСК ТАБЛИЦЫ С МОДЕЛЯМИ")
print("="*70)

model_tables = [t[0] for t in tables if 'model' in t[0].lower()]
if model_tables:
    print("✅ Найдены таблицы, связанные с моделями:")
    for t in model_tables:
        print(f"   - {t}")
else:
    print("❌ Не найдено таблиц с 'model' в названии")
    print("Возможно, таблица называется иначе. Проверьте:")
    print("   - models")
    print("   - model_versions")
    print("   - trained_models")

conn.close()