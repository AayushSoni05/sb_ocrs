from __future__ import annotations

import re
from typing import Any


def clean(
    value: str,
) -> str:

    value = value.upper()

    value = value.strip()

    value = re.sub(
        r"\s+",
        "",
        value,
    )

    return value


# =========================================================
# PORT CODE
# =========================================================

def valid_port_code(
    value: str,
) -> bool:

    value = clean(
        value
    )

    if not value:
        return False

    # Port codes on the form are short
    # alphanumeric identifiers.

    if not (
        3
        <= len(value)
        <= 10
    ):
        return False

    return bool(
        re.fullmatch(
            r"[A-Z0-9]+",
            value,
        )
    )


# =========================================================
# SHIPPING BILL NUMBER
# =========================================================

def valid_shipping_bill_number(
    value: str,
) -> bool:

    value = clean(
        value
    )

    if not value:
        return False

    # Never allow a date to become an SB number.

    if valid_date(value):
        return False

    # Must contain at least one number.

    if not re.search(
        r"\d",
        value,
    ):
        return False

    if not (
        4
        <= len(value)
        <= 20
    ):
        return False

    return bool(
        re.fullmatch(
            r"[A-Z0-9/-]+",
            value,
        )
    )


# =========================================================
# DATE
# =========================================================

def valid_date(
    value: str,
) -> bool:

    value = clean(
        value
    )

    patterns = [
        r"\d{1,2}-[A-Z]{3}-\d{2,4}",
        r"\d{1,2}/\d{1,2}/\d{2,4}",
        r"\d{1,2}-\d{1,2}-\d{2,4}",
        r"\d{1,2}\.[A-Z]{3}\.\d{2,4}",
    ]

    return any(
        re.fullmatch(
            pattern,
            value,
        )
        for pattern in patterns
    )


# =========================================================
# SELECT BEST CANDIDATE
# =========================================================

def best_candidate(
    candidates: list[dict[str, Any]],
    validator,
) -> tuple[str | None, float]:

    valid_candidates = []

    for candidate in candidates:

        text = clean(
            candidate["text"]
        )

        confidence = float(
            candidate["confidence"]
        )

        if validator(text):

            valid_candidates.append(
                (
                    text,
                    confidence,
                )
            )

    if not valid_candidates:

        return (
            None,
            0.0,
        )

    # Highest confidence.
    valid_candidates.sort(
        key=lambda item: (
            item[1],
            len(item[0]),
        ),
        reverse=True,
    )

    return valid_candidates[0]


# =========================================================
# FINAL EXTRACTION
# =========================================================

def extract_shipping_bill_fields(
    ocr_result: dict[str, Any],
) -> dict[str, Any]:

    port_code, port_confidence = (
        best_candidate(
            ocr_result.get(
                "port_code_candidates",
                [],
            ),
            valid_port_code,
        )
    )

    shipping_bill_number, sb_confidence = (
        best_candidate(
            ocr_result.get(
                "shipping_bill_candidates",
                [],
            ),
            valid_shipping_bill_number,
        )
    )

    shipping_date, date_confidence = (
        best_candidate(
            ocr_result.get(
                "shipping_date_candidates",
                [],
            ),
            valid_date,
        )
    )

    result = {
        "port_code": port_code,

        "shipping_bill_number":
            shipping_bill_number,

        "shipping_date":
            shipping_date,

        "confidence": {
            "port_code":
                round(
                    port_confidence,
                    4,
                ),

            "shipping_bill_number":
                round(
                    sb_confidence,
                    4,
                ),

            "shipping_date":
                round(
                    date_confidence,
                    4,
                ),
        },

        "needs_review": not all(
            [
                port_code,
                shipping_bill_number,
                shipping_date,
            ]
        ),
    }

    return result