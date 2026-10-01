"""
Render a PDF page to a high-resolution PNG.

The OCR pipeline works on images, so PDFs are first rendered
at high DPI and then passed into the existing OCR pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf as fitz


def pdf_page_to_png(
    pdf_path: str | Path,
    output_path: str | Path = "temp_from_pdf.png",
    page_number: int = 0,
    dpi: int = 600,
) -> Path:
    """
    Render one PDF page to PNG.

    Args:
        pdf_path: Input PDF.
        output_path: PNG output path.
        page_number: Zero-based page number.
        dpi: Rendering resolution.

    Returns:
        Path to rendered PNG.
    """

    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"File not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {pdf_path.suffix}"
        )

    doc = fitz.open(
        pdf_path
    )

    try:

        if len(doc) == 0:
            raise ValueError(
                "PDF contains no pages."
            )

        if page_number < 0 or page_number >= len(doc):
            raise ValueError(
                f"PDF has {len(doc)} page(s). "
                f"Requested page index: {page_number}"
            )

        page = doc[page_number]

        zoom = dpi / 72.0

        matrix = fitz.Matrix(
            zoom,
            zoom,
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        if output_path.parent != Path("."):
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        pix.save(
            str(output_path)
        )

        return output_path

    finally:

        doc.close()


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            r"python pdf.py samples\sb_1.pdf"
        )

        print(
            r"python pdf.py samples\sb_1.pdf 0 600"
        )

        sys.exit(1)

    pdf_path = sys.argv[1]

    page_number = (
        int(sys.argv[2])
        if len(sys.argv) > 2
        else 0
    )

    dpi = (
        int(sys.argv[3])
        if len(sys.argv) > 3
        else 600
    )

    pdf_page_to_png(
        pdf_path,
        page_number=page_number,
        dpi=dpi,
    )