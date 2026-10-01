from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv
import pg8000.dbapi

load_dotenv()

# =========================================================
# POSTGRESQL CONFIGURATION
# =========================================================

DB_HOST = os.getenv(
    "SB_DB_HOST",
    "localhost",
)

DB_PORT = int(
    os.getenv(
        "SB_DB_PORT",
        "5432",
    )
)

DB_NAME = os.getenv(
    "SB_DB_NAME",
    "ocr",
)

DB_USER = os.getenv(
    "SB_DB_USER",
    "postgres",
)

DB_PASSWORD = os.getenv(
    "SB_DB_PASSWORD",
    "",
)


# =========================================================
# CONNECTION TEST
# =========================================================

def test_connection() -> None:
    """
    Test the PostgreSQL connection.
    """

    connection = None

    try:

        connection = pg8000.dbapi.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )

        cursor = connection.cursor()

        cursor.execute(
            "SELECT current_database(), current_user;"
        )

        database_name, username = (
            cursor.fetchone()
        )

        print(
            f"Connected to database: {database_name}"
        )

        print(
            f"Connected as user: {username}"
        )

    finally:

        if connection is not None:
            connection.close()


# =========================================================
# SAVE SHIPPING BILL
# =========================================================

def save_shipping_bill(
    result: dict[str, Any],
) -> int:
    """
    Insert one extracted Shipping Bill into PostgreSQL.

    The table uses an IDENTITY column, so PostgreSQL
    generates the ID automatically.
    """

    connection = None

    try:

        connection = pg8000.dbapi.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )

        cursor = connection.cursor()

        sql = """
            INSERT INTO shipping_bill_records (
                port_code,
                shipping_bill_number,
                shipping_date
            )
            VALUES (
                %s,
                %s,
                %s
            )
            RETURNING id;
        """

        cursor.execute(
            sql,
            (
                result.get(
                    "port_code"
                ),

                result.get(
                    "shipping_bill_number"
                ),

                result.get(
                    "shipping_date"
                ),
            ),
        )

        record_id = cursor.fetchone()[0]

        connection.commit()

        return int(
            record_id
        )

    except Exception:

        if connection is not None:
            connection.rollback()

        raise

    finally:

        if connection is not None:
            connection.close()