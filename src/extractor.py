from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from PIL import Image


# Tesseract executable
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def normalize_text(text: str) -> str:
    """Basic OCR cleanup."""

    text = text.upper()

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_field(
    image: Image.Image,
) -> Image.Image:
    """
    Conservative preprocessing for a cropped field.
    """

    image = image.convert("RGB")

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY,
    )

    # Upscale the small field
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )

    # Local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    # Gentle sharpening
    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1,
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.5,
        blurred,
        -0.5,
        0,
    )

    return Image.fromarray(
        sharpened
    )


def field_ocr(
    image: Image.Image,
    whitelist: str,
    psm: int = 7,
) -> tuple[str, float]:
    """
    OCR a single field.

    psm 7 = single text line.
    """

    processed = preprocess_field(
        image
    )

    config = (
        f"--oem 3 --psm {psm} "
        f"-c tessedit_char_whitelist={whitelist} "
        "-c load_system_dawg=0 "
        "-c load_freq_dawg=0"
    )

    data = pytesseract.image_to_data(
        processed,
        lang="eng",
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    texts = []
    confidences = []

    for i, text in enumerate(
        data["text"]
    ):

        text = text.strip()

        if not text:
            continue

        texts.append(text)

        try:
            confidence = float(
                data["conf"][i]
            )

            if confidence >= 0:
                confidences.append(
                    confidence
                )

        except (
            TypeError,
            ValueError,
        ):
            pass

    result = " ".join(texts)

    confidence = (
        sum(confidences)
        / len(confidences)
        / 100
        if confidences
        else 0
    )

    return (
        result.strip(),
        round(confidence, 4),
    )


def clean_code(
    text: str,
) -> str:
    """
    Clean common OCR artifacts in identifiers.
    """

    text = normalize_text(text)

    text = text.replace(" ", "")
    text = text.replace("|", "I")

    return text


def clean_date(
    text: str,
) -> str:
    """
    Keep only date-like characters.
    """

    text = normalize_text(text)

    text = text.replace(" ", "")

    text = text.replace(",", ".")

    return text


def validate_shipping_bill_number(
    value: str,
) -> bool:
    """
    General validation for a Shipping Bill number.

    We intentionally do not assume a single exact format.
    """

    if not value:
        return False

    return bool(
        re.fullmatch(
            r"[A-Z0-9][A-Z0-9\/\-]{3,30}",
            value,
        )
    )


def validate_invoice_number(
    value: str,
) -> bool:

    if not value:
        return False

    return bool(
        re.fullmatch(
            r"[A-Z0-9][A-Z0-9\/\-_\.]{2,40}",
            value,
        )
    )


def find_label(
    words: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any] | None:
    """
    Find a label in OCR words.

    Uses normalized text and substring matching.
    """

    normalized_labels = [
        label.upper()
        for label in labels
    ]

    for word in words:

        text = normalize_text(
            word["text"]
        )

        for label in normalized_labels:

            if label in text:
                return word

    return None


def crop_right_of_label(
    image: Image.Image,
    label_word: dict[str, Any],
    right_width: int = 900,
    height: int = 180,
) -> Image.Image:
    """
    Crop the area to the right of a label.
    """

    image_width, image_height = (
        image.size
    )

    left = max(
        0,
        label_word["right"] - 10,
    )

    top = max(
        0,
        label_word["top"] - 30,
    )

    right = min(
        image_width,
        left + right_width,
    )

    bottom = min(
        image_height,
        top + height,
    )

    return image.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )


def crop_below_label(
    image: Image.Image,
    label_word: dict[str, Any],
    width: int = 900,
    height: int = 220,
) -> Image.Image:
    """
    Crop the area below a label.
    """

    image_width, image_height = (
        image.size
    )

    left = max(
        0,
        label_word["left"] - 30,
    )

    top = label_word["bottom"]

    right = min(
        image_width,
        left + width,
    )

    bottom = min(
        image_height,
        top + height,
    )

    return image.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )


def extract_shipping_bill_fields(
    image_path: str | Path,
    ocr_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract:
        - Shipping Bill Number
        - Shipping Date
        - Invoice Number

    from OCR layout information.
    """

    image_path = Path(
        image_path
    )

    image = Image.open(
        image_path
    )

    words = ocr_result.get(
        "words",
        [],
    )

    result = {
        "shipping_bill_number": None,
        "shipping_date": None,
        "invoice_number": None,

        "confidence": {
            "shipping_bill_number": 0.0,
            "shipping_date": 0.0,
            "invoice_number": 0.0,
        },

        "needs_review": True,
    }

    # -------------------------------------------------
    # SHIPPING BILL NUMBER
    # -------------------------------------------------

    shipping_label = find_label(
        words,
        [
            "SHIPPING BILL NO",
            "SHIPPING BILL NUMBER",
            "SB NO",
            "SB NUMBER",
            "SHIPPING BILL",
        ],
    )

    if shipping_label:

        crops = [
            crop_right_of_label(
                image,
                shipping_label,
            ),
            crop_below_label(
                image,
                shipping_label,
            ),
        ]

        candidates = []

        for crop in crops:

            text, confidence = field_ocr(
                crop,
                whitelist=(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789/-"
                ),
                psm=7,
            )

            value = clean_code(
                text
            )

            if validate_shipping_bill_number(
                value
            ):
                candidates.append(
                    (
                        value,
                        confidence,
                    )
                )

        if candidates:

            best = max(
                candidates,
                key=lambda item: item[1],
            )

            result[
                "shipping_bill_number"
            ] = best[0]

            result[
                "confidence"
            ][
                "shipping_bill_number"
            ] = best[1]

    # -------------------------------------------------
    # SHIPPING DATE
    # -------------------------------------------------

    date_label = find_label(
        words,
        [
            "SB DATE",
            "SHIPPING DATE",
            "DATE OF SHIPMENT",
            "SHIPPED ON BOARD",
        ],
    )

    if date_label:

        crops = [
            crop_right_of_label(
                image,
                date_label,
            ),
            crop_below_label(
                image,
                date_label,
            ),
        ]

        candidates = []

        for crop in crops:

            text, confidence = field_ocr(
                crop,
                whitelist=(
                    "0123456789"
                    "/-.:"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                ),
                psm=7,
            )

            value = clean_date(
                text
            )

            if re.search(
                r"\d{1,2}.*\d{1,2}.*\d{2,4}",
                value,
            ):
                candidates.append(
                    (
                        value,
                        confidence,
                    )
                )

        if candidates:

            best = max(
                candidates,
                key=lambda item: item[1],
            )

            result[
                "shipping_date"
            ] = best[0]

            result[
                "confidence"
            ][
                "shipping_date"
            ] = best[1]

    # -------------------------------------------------
    # INVOICE NUMBER
    # -------------------------------------------------

    invoice_label = find_label(
        words,
        [
            "INVOICE NO",
            "INVOICE NUMBER",
            "INV NO",
            "INVOICE",
        ],
    )

    if invoice_label:

        crops = [
            crop_right_of_label(
                image,
                invoice_label,
            ),
            crop_below_label(
                image,
                invoice_label,
            ),
        ]

        candidates = []

        for crop in crops:

            text, confidence = field_ocr(
                crop,
                whitelist=(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789/-_."
                ),
                psm=7,
            )

            value = clean_code(
                text
            )

            if validate_invoice_number(
                value
            ):
                candidates.append(
                    (
                        value,
                        confidence,
                    )
                )

        if candidates:

            best = max(
                candidates,
                key=lambda item: item[1],
            )

            result[
                "invoice_number"
            ] = best[0]

            result[
                "confidence"
            ][
                "invoice_number"
            ] = best[1]

    # -------------------------------------------------
    # REVIEW DECISION
    # -------------------------------------------------

    field_confidences = result[
        "confidence"
    ]

    valid_fields = 0

    for confidence in field_confidences.values():

        if confidence >= 0.80:
            valid_fields += 1

    result[
        "needs_review"
    ] = valid_fields < 2

    return result