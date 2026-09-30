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

        if candidate.get("low_resolution"):
            continue

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
# FALLBACK: PARSE THE WHOLE VALUE ROW TEXT
# =========================================================
#
# Used when the cell-based approach fails/has 0 confidence -
# typically on low-resolution images where each individual
# cell was too small for Tesseract to read reliably. Here we
# work from ONE OCR pass over the whole row instead, and pick
# out the 3 values by what they look like (a date pattern, an
# identifier with a digit, etc) rather than by position.

def parse_whole_row_candidates(
    candidates: list[dict[str, Any]],
) -> dict[str, str | None]:

    for candidate in candidates:

        text = candidate.get(
            "text",
            "",
        )

        tokens = [
            clean(token)
            for token in text.split()
            if clean(token)
        ]

        if not tokens:
            continue

        date_token = None
        remaining = []

        for token in tokens:

            if date_token is None and valid_date(token):
                date_token = token
            else:
                remaining.append(token)

        port_token = None
        sb_token = None

        for token in remaining:

            if port_token is None and valid_port_code(token):
                port_token = token
            elif sb_token is None and valid_shipping_bill_number(token):
                sb_token = token

        # Only accept this candidate's parse if we found at
        # least one usable field from it - a completely empty
        # parse isn't worth returning.

        if port_token or sb_token or date_token:

            return {
                "port_code": port_token,
                "shipping_bill_number": sb_token,
                "shipping_date": date_token,
            }

    return {
        "port_code": None,
        "shipping_bill_number": None,
        "shipping_date": None,
    }


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

    # ---------------------------------------------------
    # Which fields came from a cell too small to trust at
    # all (see MIN_READABLE_HEIGHT_PX in ocr.py)? For those,
    # don't even trust the whole-row fallback guess - the
    # underlying pixels are the same low-res source, so a
    # different OCR pass over them just produces a different
    # wrong answer, not a real one. Leave these None so they
    # surface as needs_review / manual entry instead of
    # silently storing a plausible-looking wrong value.
    # ---------------------------------------------------

    low_res_fields = [
        field
        for field, candidates in [
            ("port_code", ocr_result.get("port_code_candidates", [])),
            ("shipping_bill_number", ocr_result.get("shipping_bill_candidates", [])),
            ("shipping_date", ocr_result.get("shipping_date_candidates", [])),
        ]
        if any(c.get("low_resolution") for c in candidates)
    ]

    # ---------------------------------------------------
    # Fallback pass: for any field that came back empty or
    # zero-confidence (but was NOT flagged low-resolution),
    # try the whole-row parse instead.
    # ---------------------------------------------------

    used_fallback = False

    needs_fallback = (
        not port_code
        or not shipping_bill_number
        or not shipping_date
        or min(
            port_confidence,
            sb_confidence,
            date_confidence,
        )
        <= 0.0
    )

    if needs_fallback:

        fallback = parse_whole_row_candidates(
            ocr_result.get(
                "whole_row_candidates",
                [],
            )
        )

        if not port_code and fallback["port_code"] and "port_code" not in low_res_fields:
            port_code = fallback["port_code"]
            used_fallback = True

        if (
            not shipping_bill_number
            and fallback["shipping_bill_number"]
            and "shipping_bill_number" not in low_res_fields
        ):
            shipping_bill_number = fallback["shipping_bill_number"]
            used_fallback = True

        if (
            not shipping_date
            and fallback["shipping_date"]
            and "shipping_date" not in low_res_fields
        ):
            shipping_date = fallback["shipping_date"]
            used_fallback = True

    # Belt-and-suspenders: make sure low-res fields are None
    # in the final output even if something upstream slipped
    # a value through.
    if "port_code" in low_res_fields:
        port_code = None
    if "shipping_bill_number" in low_res_fields:
        shipping_bill_number = None
    if "shipping_date" in low_res_fields:
        shipping_date = None

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

        "used_whole_row_fallback": used_fallback,

        "low_resolution_fields": low_res_fields,

        "needs_review": not all(
            [
                port_code,
                shipping_bill_number,
                shipping_date,
            ]
        ),
    }

    return result