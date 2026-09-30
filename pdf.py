"""
Convert a PDF page to a high-resolution PNG so main.py can run on it.

Shipping bill PDFs are usually vector/text-based and render at whatever
DPI we ask for - unlike a screenshot, we're not stuck with a fixed low
resolution. 300 DPI gives Tesseract plenty of real pixels to work with.
"""
import sys
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (the `fitz` import name is deprecated)


def pdf_page_to_png(
    pdf_path: str,
    output_path: str = "output/from_pdf.png",
    page_number: int = 0,
    dpi: int = 300,
):
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    if page_number >= len(doc):
        raise ValueError(
            f"PDF only has {len(doc)} page(s), "
            f"page {page_number} doesn't exist."
        )

    page = doc[page_number]

    # zoom factor: PyMuPDF's default render is 72 DPI, so scale up
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=matrix)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pix.save(output_path)

    print(f"Rendered page {page_number} at {dpi} DPI -> {output_path}")
    print(f"Image size: {pix.width}x{pix.height} px")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_image.py <path-to.pdf> [page_number] [dpi]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_number = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    pdf_page_to_png(pdf_path, page_number=page_number, dpi=dpi)