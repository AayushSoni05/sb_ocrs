from pathlib import Path
import json
import re

import cv2
import numpy as np
import pytesseract


# ---------------------------------------------------------
# TESSERACT
# ---------------------------------------------------------

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


INPUT_IMAGE = Path(
    "output/header_crop.png"
)

OUTPUT_DIR = Path(
    "output"
)


# =========================================================
# LOAD
# =========================================================

def load_image(
    path: Path,
) -> np.ndarray:

    image = cv2.imread(
        str(path)
    )

    if image is None:

        raise ValueError(
            f"Could not read image: {path}"
        )

    return image


# =========================================================
# CROP VALUE ROW
# =========================================================

def crop_value_row(
    header: np.ndarray,
) -> np.ndarray:

    height, width = header.shape[:2]

    # Looking at your actual header crop:
    #
    # green field labels occupy the upper part
    # of the table.
    #
    # The first actual value row is directly below them.

    y1 = int(
        height * 0.25
    )

    y2 = int(
        height * 0.43
    )

    return header[
        y1:y2,
        :
    ]


# =========================================================
# SPLIT THREE VALUE CELLS
# =========================================================

def split_cells(
    value_row: np.ndarray,
) -> dict[str, np.ndarray]:

    height, width = (
        value_row.shape[:2]
    )

    # Based on the visible vertical table lines
    # in your uploaded crop.

    port_x1 = 0
    port_x2 = int(
        width * 0.31
    )

    sb_x1 = int(
        width * 0.31
    )

    sb_x2 = int(
        width * 0.69
    )

    date_x1 = int(
        width * 0.69
    )

    date_x2 = width

    return {

        "port_code": value_row[
            :,
            port_x1:port_x2,
        ],

        "shipping_bill_number": value_row[
            :,
            sb_x1:sb_x2,
        ],

        "shipping_date": value_row[
            :,
            date_x1:date_x2,
        ],
    }


# =========================================================
# PREPROCESS
# =========================================================

def preprocess_variants(
    image: np.ndarray,
) -> list[np.ndarray]:

    # -----------------------------------------------------
    # Upscale heavily because the source characters
    # are extremely small.
    # -----------------------------------------------------

    enlarged = cv2.resize(
        image,
        None,
        fx=10,
        fy=10,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY,
    )

    # -----------------------------------------------------
    # Contrast
    # -----------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    contrast = clahe.apply(
        gray
    )

    # -----------------------------------------------------
    # Threshold
    # -----------------------------------------------------

    _, otsu = cv2.threshold(
        contrast,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    # -----------------------------------------------------
    # Adaptive
    # -----------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    # -----------------------------------------------------
    # Sharpen
    # -----------------------------------------------------

    blurred = cv2.GaussianBlur(
        contrast,
        (0, 0),
        1.0,
    )

    sharpened = cv2.addWeighted(
        contrast,
        1.8,
        blurred,
        -0.8,
        0,
    )

    return [
        enlarged,
        contrast,
        otsu,
        adaptive,
        sharpened,
    ]


# =========================================================
# OCR
# =========================================================

def run_ocr(
    image: np.ndarray,
    whitelist: str,
) -> list[dict]:

    candidates = []

    for variant_number, variant in enumerate(
        preprocess_variants(image),
        start=1,
    ):

        for psm in [
            6,
            7,
            8,
            13,
        ]:

            config = (
                f"--oem 3 --psm {psm} "
                f"-c tessedit_char_whitelist={whitelist} "
                "-c preserve_interword_spaces=1"
            )

            data = pytesseract.image_to_data(
                variant,
                lang="eng",
                config=config,
                output_type=pytesseract.Output.DICT,
            )

            pieces = []
            confidence_values = []

            for i, text in enumerate(
                data["text"]
            ):

                text = text.strip()

                if not text:
                    continue

                try:

                    confidence = float(
                        data["conf"][i]
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    confidence = -1

                if confidence < 0:
                    continue

                pieces.append(
                    text
                )

                confidence_values.append(
                    confidence
                )

            if not pieces:
                continue

            combined = "".join(
                pieces
            )

            average_confidence = (
                sum(confidence_values)
                / len(confidence_values)
                / 100
            )

            candidates.append(
                {
                    "variant": variant_number,
                    "psm": psm,
                    "text": combined,
                    "confidence": round(
                        average_confidence,
                        4,
                    ),
                }
            )

    return candidates


# =========================================================
# DATE VALIDATION
# =========================================================

def clean_date(
    value: str,
) -> str:

    value = value.upper()

    value = re.sub(
        r"\s+",
        "",
        value,
    )

    # Common OCR substitutions
    value = value.replace(
        "0",
        "0",
    )

    return value


def valid_date(
    value: str,
) -> bool:

    value = clean_date(
        value
    )

    patterns = [
        r"^\d{1,2}-[A-Z]{3}-\d{2,4}$",
        r"^\d{1,2}/\d{1,2}/\d{2,4}$",
        r"^\d{1,2}-\d{1,2}-\d{2,4}$",
    ]

    return any(
        re.fullmatch(
            pattern,
            value,
        )
        for pattern in patterns
    )


# =========================================================
# IDENTIFIER VALIDATION
# =========================================================

def valid_identifier(
    value: str,
) -> bool:

    value = value.upper()

    value = re.sub(
        r"[^A-Z0-9]",
        "",
        value,
    )

    if len(value) < 3:
        return False

    # It cannot be a date.
    if valid_date(value):
        return False

    # Must contain at least one digit.
    if not re.search(
        r"\d",
        value,
    ):
        return False

    return True


# =========================================================
# CHOOSE BEST IDENTIFIER
# =========================================================

def choose_identifier(
    candidates: list[dict],
) -> tuple[str | None, float]:

    valid = []

    for candidate in candidates:

        value = re.sub(
            r"[^A-Z0-9/-]",
            "",
            candidate["text"].upper(),
        )

        if not valid_identifier(
            value
        ):
            continue

        valid.append(
            (
                value,
                candidate["confidence"],
            )
        )

    if not valid:

        return (
            None,
            0.0,
        )

    # Highest OCR confidence,
    # then longest meaningful value.

    valid.sort(
        key=lambda x: (
            x[1],
            len(x[0]),
        ),
        reverse=True,
    )

    return valid[0]


# =========================================================
# CHOOSE DATE
# =========================================================

def choose_date(
    candidates: list[dict],
) -> tuple[str | None, float]:

    valid = []

    for candidate in candidates:

        value = clean_date(
            candidate["text"]
        )

        if valid_date(
            value
        ):

            valid.append(
                (
                    value,
                    candidate["confidence"],
                )
            )

    if not valid:

        return (
            None,
            0.0,
        )

    valid.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return valid[0]


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = load_image(
        INPUT_IMAGE
    )

    print(
        f"Header image: {INPUT_IMAGE}"
    )

    # -----------------------------------------------------
    # Value row
    # -----------------------------------------------------

    value_row = crop_value_row(
        header
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "debug_value_row.png"
        ),
        value_row,
    )

    # -----------------------------------------------------
    # Cells
    # -----------------------------------------------------

    cells = split_cells(
        value_row
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "debug_port_code.png"
        ),
        cells["port_code"],
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "debug_shipping_bill.png"
        ),
        cells[
            "shipping_bill_number"
        ],
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "debug_shipping_date.png"
        ),
        cells["shipping_date"],
    )

    # -----------------------------------------------------
    # OCR
    # -----------------------------------------------------

    port_candidates = run_ocr(
        cells["port_code"],
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )

    sb_candidates = run_ocr(
        cells[
            "shipping_bill_number"
        ],
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-",
    )

    date_candidates = run_ocr(
        cells["shipping_date"],
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-",
    )

    # -----------------------------------------------------
    # Print candidates
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("PORT CODE CANDIDATES")
    print("=" * 70)

    for candidate in port_candidates:
        print(candidate)

    print()
    print("=" * 70)
    print("SHIPPING BILL CANDIDATES")
    print("=" * 70)

    for candidate in sb_candidates:
        print(candidate)

    print()
    print("=" * 70)
    print("DATE CANDIDATES")
    print("=" * 70)

    for candidate in date_candidates:
        print(candidate)

    # -----------------------------------------------------
    # Final values
    # -----------------------------------------------------

    port_code, port_confidence = (
        choose_identifier(
            port_candidates
        )
    )

    shipping_bill_number, sb_confidence = (
        choose_identifier(
            sb_candidates
        )
    )

    shipping_date, date_confidence = (
        choose_date(
            date_candidates
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
                port_confidence,

            "shipping_bill_number":
                sb_confidence,

            "shipping_date":
                date_confidence,
        },

        "needs_review": not all(
            [
                port_code,
                shipping_bill_number,
                shipping_date,
            ]
        ),
    }

    # -----------------------------------------------------
    # Print result
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL JSON")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=4,
        )
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "shipping_bill_result.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print(
        f"Saved: {output_file}"
    )


if __name__ == "__main__":
    main()