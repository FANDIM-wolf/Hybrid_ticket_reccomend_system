"""
Запуск микросервиса панели администратора.

Использование:
    python admin_start.py                     # порт 8001, дефолтные credentials
    python admin_start.py --port 9000         # кастомный порт

    ADMIN_USERNAME=root ADMIN_PASSWORD=secret python admin_start.py

Требует работающего основного приложения на порту 8000
(для проксирования мутирующих операций).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("admin_start")


def main():
    parser = argparse.ArgumentParser(description="DSS Admin Panel")
    parser.add_argument("--port", type=int, default=8001, help="Admin port (default: 8001)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Admin host")
    args = parser.parse_args()

    import uvicorn

    logger.info("=" * 60)
    logger.info("DSS Admin Panel starting")
    logger.info(f"Admin:      http://{args.host}:{args.port}")
    logger.info(f"Admin docs: http://{args.host}:{args.port}/docs")
    logger.info(f"Main app:   http://127.0.0.1:8000")
    logger.info("=" * 60)
    logger.info("Default credentials: admin / dss_admin_2026")
    logger.info("Override via env: ADMIN_USERNAME, ADMIN_PASSWORD")
    logger.info("=" * 60)

    uvicorn.run(
        "admin.admin_app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
