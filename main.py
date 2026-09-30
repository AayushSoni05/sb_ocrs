"""
Entry point: full shipping bill image -> port_code, shipping_bill_number,
shipping_date (+ confidence, needs_review) -> printed + saved as JSON.

Wires together src/ocr.py (crop + OCR the header) and src/extractor.py
(validate + pick best candidate per field).
"""
import json
import sys
from pathlib import Path

from src.ocr import ShippingBillOCR
from src.extractor import extract_shipping_bill_fields

OUTPUT_DIR = Path("output")


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <path-to-shipping-bill-image>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Input image: {image_path}")
    print("Running OCR (this crops the header, splits it into 3 cells, "
          "and tries several preprocessing variants per cell)...")

    ocr = ShippingBillOCR()
    ocr_result = ocr.extract(image_path)

    print()
    print("=" * 70)
    print("PORT CODE CANDIDATES")
    print("=" * 70)
    for candidate in ocr_result["port_code_candidates"]:
        print(candidate)

    print()
    print("=" * 70)
    print("SHIPPING BILL CANDIDATES")
    print("=" * 70)
    for candidate in ocr_result["shipping_bill_candidates"]:
        print(candidate)

    print()
    print("=" * 70)
    print("DATE CANDIDATES")
    print("=" * 70)
    for candidate in ocr_result["shipping_date_candidates"]:
        print(candidate)

    print()
    print("=" * 70)
    print("WHOLE-ROW CANDIDATES (fallback)")
    print("=" * 70)
    for candidate in ocr_result["whole_row_candidates"]:
        print(candidate)

    fields = extract_shipping_bill_fields(ocr_result)

    print()
    print("=" * 70)
    print("FINAL JSON")
    print("=" * 70)
    print(json.dumps(fields, indent=4))

    output_file = OUTPUT_DIR / "shipping_bill_result.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(fields, f, indent=4, ensure_ascii=False)

    print()
    print(f"Saved: {output_file}")

    if fields["needs_review"]:
        print()
        print("NOTE: needs_review is True - at least one field could not be "
              "confidently extracted. Check output/port_cell.png, "
              "output/shipping_bill_cell.png, output/date_cell.png to see "
              "what Tesseract was actually looking at.")


if __name__ == "__main__":
    main()