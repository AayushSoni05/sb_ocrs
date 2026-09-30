from pathlib import Path
import sys

from src.ocr import extract_text


def main() -> None:

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <file>")
        print()
        print("Example:")
        print("  python main.py samples\\shipping_bill.pdf")
        return

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    print("=" * 70)
    print("SHIPPING BILL OCR")
    print("=" * 70)

    print(f"File: {file_path}")

    try:
        result = extract_text(file_path)
        from src.extractor import extract_shipping_bill_fields

    except Exception as error:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(error)
        return

    # -------------------------------------------------
    # Main OCR output
    # -------------------------------------------------

    print()
    print("=" * 70)
    print("EXTRACTED TEXT")
    print("=" * 70)

    print(result.get("text", ""))

    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    print()
    print("=" * 70)
    print("METADATA")
    print("=" * 70)

    if "pages" in result:

        for page in result["pages"]:

            print(
                f"Page {page['page']} | "
                f"Method: {page['method']} | "
                f"Confidence: {page['confidence']}"
            )

            # Show OCR variants for OCR pages
            if "variants" in page:

                print()
                print("OCR VARIANTS")

                for variant in page["variants"]:

                    print()
                    print(
                        f"VARIANT: {variant['variant']} | "
                        f"CONFIDENCE: {variant['confidence']}"
                    )

                    print("-" * 70)
                    print(variant["text"])

    else:

        print(
            f"Type: {result.get('type', 'unknown')}"
        )

        print(
            f"Confidence: {result.get('confidence', 0)}"
        )

        # Show OCR variants for image files
        if "variants" in result:

            print()
            print("=" * 70)
            print("OCR VARIANTS")
            print("=" * 70)

            for variant in result["variants"]:

                print()
                print(
                    f"PSM: {variant['psm']} | "
                    f"CONFIDENCE: {variant['confidence']}"
                )

                print("-" * 70)

                print(
                    variant["text"]
                )


if __name__ == "__main__":
    main()