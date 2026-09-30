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
) -> dict[str, Any]:
    """
    Parse a complete:

        PORT CODE
        SHIPPING BILL NUMBER
        SHIPPING DATE

    triplet from whole-row OCR output.

    Example accepted row:

        INSBI6 4874706 09-JUL-26

    We require all three values to occur together.
    """

    triplets = []

    # ---------------------------------------------------------
    # Expected pattern
    #
    # PORT CODE:
    #     3-10 alphanumeric characters
    #
    # SB NUMBER:
    #     4-20 characters containing digits
    #
    # DATE:
    #     DD-MMM-YY / DD-MMM-YYYY
    # ---------------------------------------------------------

    pattern = re.compile(
        r"""
        (?P<port>
            [A-Z0-9]{3,10}
        )

        \s+

        (?P<sb>
            [A-Z0-9/-]{4,20}
        )

        \s+

        (?P<date>
            \d{1,2}
            -
            [A-Z]{3}
            -
            \d{2,4}
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    for candidate in candidates:

        text = candidate.get(
            "text",
            "",
        )

        if not text:
            continue

        # OCR may put a leading '-' before the first value.
        text = text.strip()

        matches = pattern.finditer(
            text
        )

        for match in matches:

            port = clean(
                match.group("port")
            )

            sb = clean(
                match.group("sb")
            )

            date = clean(
                match.group("date")
            )

            # -------------------------------------------------
            # Validate each part
            # -------------------------------------------------

            if not valid_port_code(
                port
            ):
                continue

            if not valid_shipping_bill_number(
                sb
            ):
                continue

            if not valid_date(
                date
            ):
                continue

            triplets.append(
                {
                    "port_code": port,
                    "shipping_bill_number": sb,
                    "shipping_date": date,
                    "variant": candidate.get(
                        "variant"
                    ),
                }
            )

    # ---------------------------------------------------------
    # Nothing found
    # ---------------------------------------------------------

    if not triplets:

        return {
            "port_code": None,
            "shipping_bill_number": None,
            "shipping_date": None,
            "consensus_count": 0,
        }

    # ---------------------------------------------------------
    # Find repeated triplets
    # ---------------------------------------------------------

    counts: dict[
        tuple[str, str, str],
        int
    ] = {}

    for item in triplets:

        key = (
            item["port_code"],
            item["shipping_bill_number"],
            item["shipping_date"],
        )

        counts[key] = (
            counts.get(
                key,
                0,
            )
            + 1
        )

    # Highest consensus.
    best_key = max(
        counts,
        key=counts.get,
    )

    consensus_count = counts[
        best_key
    ]

    return {
        "port_code": best_key[0],

        "shipping_bill_number":
            best_key[1],

        "shipping_date":
            best_key[2],

        "consensus_count":
            consensus_count,
    }

# =========================================================
# FINAL EXTRACTION
# =========================================================

def extract_shipping_bill_fields(
    ocr_result: dict[str, Any],
) -> dict[str, Any]:

    # =========================================================
    # FIRST: LOOK FOR A COMPLETE WHOLE-ROW MATCH
    # =========================================================

    whole_row = parse_whole_row_candidates(
        ocr_result.get(
            "whole_row_candidates",
            [],
        )
    )

    # =========================================================
    # PRIMARY PATH
    # =========================================================
    #
    # If at least 2 OCR variants independently produce
    # the SAME complete triplet, use that triplet.
    #
    # Example:
    #
    # Variant 1:
    #     INSBI6 4874706 09-JUL-26
    #
    # Variant 2:
    #     INSBI6 4874706 09-JUL-26
    #
    # Variant 3:
    #     INSBI6 4874706 09-JUL-26
    #
    # This is much stronger than accepting one bad
    # individual-cell result.
    # =========================================================

    if whole_row["consensus_count"] >= 2:

        result = {
            "port_code":
                whole_row["port_code"],

            "shipping_bill_number":
                whole_row[
                    "shipping_bill_number"
                ],

            "shipping_date":
                whole_row[
                    "shipping_date"
                ],

            "confidence": {
                # These are left at 0 because the whole-row
                # OCR path currently uses image_to_string()
                # and does not provide word confidence.
                #
                # Do NOT pretend that consensus count is
                # Tesseract confidence.
                "port_code": 0.0,

                "shipping_bill_number": 0.0,

                "shipping_date": 0.0,
            },

            "used_whole_row_fallback": True,

            "whole_row_consensus_count":
                whole_row[
                    "consensus_count"
                ],

            "extraction_source":
                "whole_row_consensus",

            # We have a strong extraction signal, but we
            # don't yet have actual OCR confidence for this
            # path.
            "needs_review": False,
        }

        return result

    # =========================================================
    # SECONDARY PATH: INDIVIDUAL CELLS
    # =========================================================

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

    # =========================================================
    # LOW-RESOLUTION FIELDS
    # =========================================================

    low_res_fields = [

        field

        for field, candidates in [

            (
                "port_code",
                ocr_result.get(
                    "port_code_candidates",
                    [],
                ),
            ),

            (
                "shipping_bill_number",
                ocr_result.get(
                    "shipping_bill_candidates",
                    [],
                ),
            ),

            (
                "shipping_date",
                ocr_result.get(
                    "shipping_date_candidates",
                    [],
                ),
            ),
        ]

        if any(
            candidate.get(
                "low_resolution"
            )
            for candidate in candidates
        )
    ]

    # =========================================================
    # FINAL SECONDARY RESULT
    # =========================================================

    result = {
        "port_code":
            port_code,

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

        "used_whole_row_fallback":
            False,

        "whole_row_consensus_count":
            whole_row[
                "consensus_count"
            ],

        "extraction_source":
            "individual_cells",

        "low_resolution_fields":
            low_res_fields,

        "needs_review": not (
            port_code
            and shipping_bill_number
            and shipping_date
            and port_confidence >= 0.80
            and sb_confidence >= 0.80
            and date_confidence >= 0.80
        ),
    }

    return result