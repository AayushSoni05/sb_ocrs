from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from src.ocr import ShippingBillOCR
from src.extractor import extract_shipping_bill_fields
from pdf import pdf_page_to_png
from src.database import save_shipping_bill


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


def prepare_input(
    input_path: Path,
) -> tuple[Path, bool]:

    extension = input_path.suffix.lower()

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

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False,
        )

        rendered_path = Path(
            temp_file.name
        )

        temp_file.close()

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

    raise ValueError(
        "Unsupported file type: "
        f"{extension}"
    )


def print_json(
    data: dict,
) -> None:

    print(
        json.dumps(
            data,
            indent=4,
            ensure_ascii=False,
        )
    )


def main() -> None:

    # -------------------------------------------------
    # ARGUMENT
    # -------------------------------------------------

    if len(sys.argv) != 2:

        print_json(
            {
                "error": (
                    "Usage: "
                    "python main.py <PDF_or_image>"
                )
            }
        )

        sys.exit(1)

    input_path = Path(
        sys.argv[1]
    )

    if not input_path.exists():

        print_json(
            {
                "error": (
                    f"File not found: {input_path}"
                )
            }
        )

        sys.exit(1)

    rendered_path = None
    came_from_pdf = False

    try:

        # -------------------------------------------------
        # PREPARE INPUT
        # -------------------------------------------------

        image_path, came_from_pdf = (
            prepare_input(
                input_path
            )
        )

        rendered_path = (
            image_path
            if came_from_pdf
            else None
        )

        # -------------------------------------------------
        # OCR
        # -------------------------------------------------

        ocr = ShippingBillOCR()

        ocr_result = ocr.extract(
            image_path
        )

        # -------------------------------------------------
        # EXTRACTION
        # -------------------------------------------------

        fields = extract_shipping_bill_fields(
            ocr_result
        )

        # -------------------------------------------------
        # DATABASE
        # -------------------------------------------------

        if (
            not fields.get(
                "needs_review",
                True,
            )
            and fields.get(
                "port_code"
            )
            and fields.get(
                "shipping_bill_number"
            )
            and fields.get(
                "shipping_date"
            )
        ):

            try:

                record_id = save_shipping_bill(
                    fields
                )

                fields["database_id"] = record_id

            except Exception as error:

                fields["database_error"] = str(error)

        # -------------------------------------------------
        # FINAL JSON ONLY
        # -------------------------------------------------

        print_json(
            fields
)

    except Exception as error:

        print_json(
            {
                "error": str(error)
            }
        )

        sys.exit(1)

    finally:

        # -------------------------------------------------
        # DELETE TEMPORARY PDF RENDER
        # -------------------------------------------------

        if rendered_path is not None:

            rendered_path.unlink(
                missing_ok=True
            )


if __name__ == "__main__":
    main()