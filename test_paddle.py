from pathlib import Path

from paddleocr import PaddleOCR


IMAGE_PATH = Path("C:\\Users\\Admin\\Downloads\\image (1).png")
OUTPUT_DIR = Path("sb2.png")


def main() -> None:

    if not IMAGE_PATH.exists():
        print(f"File not found: {IMAGE_PATH}")
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("PADDLEOCR SHIPPING BILL TEST")
    print("=" * 70)

    print(f"Input: {IMAGE_PATH}")

    # Document preprocessing is enabled here.
    # This allows orientation detection and unwarping.
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
    )

    print()
    print("Running PaddleOCR...")

    results = ocr.predict(
        str(IMAGE_PATH)
    )

    for result in results:

        result.print()

        result.save_to_img(
            str(OUTPUT_DIR)
        )

        result.save_to_json(
            str(OUTPUT_DIR)
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Results saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()