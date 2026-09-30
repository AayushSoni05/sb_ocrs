"""
Shipping Bill extraction pipeline.

Supported input:

    PNG
    JPG
    JPEG
    TIFF
    PDF

For PDFs:
    PDF -> 600 DPI PNG -> OCR -> extraction -> JSON

For images:
    Image -> OCR -> extraction -> JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.ocr import ShippingBillOCR
from src.extractor import extract_shipping_bill_fields
from pdf import pdf_page_to_png


OUTPUT_DIR = Path(
    "output"
)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


def print_section(
    title: str,
) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def prepare_input(
    input_path: Path,
) -> tuple[Path, bool]:

    extension = (
        input_path.suffix.lower()
    )

    # -------------------------------------------------
    # IMAGE
    # -------------------------------------------------

    if extension in SUPPORTED_IMAGE_EXTENSIONS:

        return (
            input_path,
            False,
        )

    # -------------------------------------------------
    # PDF
    # -------------------------------------------------

    if extension == ".pdf":

        rendered_path = (
            OUTPUT_DIR
            / "from_pdf.png"
        )

        pdf_page_to_png(
            input_path,
            output_path=rendered_path,
            page_number=0,
            dpi=600,
        )

        return (
            rendered_path,
            True,
        )

    # -------------------------------------------------
    # Unsupported
    # -------------------------------------------------

    raise ValueError(
        "Unsupported file type: "
        f"{extension}\n"
        "Supported: PDF, PNG, JPG, JPEG, TIF, TIFF"
    )


def main() -> None:

    # =================================================
    # ARGUMENT
    # =================================================

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            r"  python main.py samples\sb_1.png"
        )

        print(
            r"  python main.py samples\sb_1.pdf"
        )

        sys.exit(1)

    input_path = Path(
        sys.argv[1]
    )

    if not input_path.exists():

        print(
            f"File not found: {input_path}"
        )

        sys.exit(1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_section(
        "SHIPPING BILL EXTRACTION"
    )

    print(
        f"Input: {input_path}"
    )

    # =================================================
    # PREPARE INPUT
    # =================================================

    try:

        image_path, came_from_pdf = (
            prepare_input(
                input_path
            )
        )

    except Exception as error:

        print_section(
            "INPUT ERROR"
        )

        print(
            repr(error)
        )

        sys.exit(1)

    # =================================================
    # OCR
    # =================================================

    print_section(
        "OCR"
    )

    print(
        f"OCR image: {image_path}"
    )

    if came_from_pdf:

        print(
            "Source: PDF rendered at 600 DPI"
        )

    else:

        print(
            "Source: original image"
        )

    try:

        ocr = ShippingBillOCR()

        ocr_result = ocr.extract(
            image_path
        )

    except Exception as error:

        print_section(
            "OCR ERROR"
        )

        print(
            repr(error)
        )

        sys.exit(1)

    # =================================================
    # OCR CANDIDATES
    # =================================================

    print_section(
        "PORT CODE CANDIDATES"
    )

    for candidate in ocr_result.get(
        "port_code_candidates",
        [],
    ):

        print(
            candidate
        )

    print_section(
        "SHIPPING BILL CANDIDATES"
    )

    for candidate in ocr_result.get(
        "shipping_bill_candidates",
        [],
    ):

        print(
            candidate
        )

    print_section(
        "DATE CANDIDATES"
    )

    for candidate in ocr_result.get(
        "shipping_date_candidates",
        [],
    ):

        print(
            candidate
        )

    print_section(
        "WHOLE ROW CANDIDATES"
    )

    for candidate in ocr_result.get(
        "whole_row_candidates",
        [],
    ):

        print(
            candidate
        )

    # =================================================
    # EXTRACTION
    # =================================================

    print_section(
        "STRUCTURED EXTRACTION"
    )

    try:

        fields = (
            extract_shipping_bill_fields(
                ocr_result
            )
        )

    except Exception as error:

        print_section(
            "EXTRACTION ERROR"
        )

        print(
            repr(error)
        )

        sys.exit(1)

    # =================================================
    # FINAL JSON
    # =================================================

    print_section(
        "FINAL JSON"
    )

    print(
        json.dumps(
            fields,
            indent=4,
            ensure_ascii=False,
        )
    )

    # =================================================
    # SAVE JSON
    # =================================================

    output_file = (
        OUTPUT_DIR
        / "shipping_bill_result.json"
    )

    try:

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                fields,
                file,
                indent=4,
                ensure_ascii=False,
            )

    except Exception as error:

        print_section(
            "JSON SAVE ERROR"
        )

        print(
            repr(error)
        )

        sys.exit(1)

    print()
    print(
        f"JSON saved to: {output_file}"
    )

    # =================================================
    # REVIEW
    # =================================================

    print()

    if fields.get(
        "needs_review",
        True,
    ):

        print(
            "REVIEW STATUS: "
            "MANUAL REVIEW REQUIRED"
        )

    else:

        print(
            "REVIEW STATUS: "
            "ALL REQUIRED FIELDS EXTRACTED"
        )


if __name__ == "__main__":
    main()