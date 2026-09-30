from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf
import pytesseract
from PIL import Image


# Windows Tesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}


def prepare_image(
    image: Image.Image,
) -> np.ndarray:
    """
    Conservative preprocessing.

    We avoid aggressive thresholding because it can break
    characters in poor-quality photos.
    """

    image = image.convert("RGB")

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY,
    )

    # 2x upscale
    gray = cv2.resize(
        gray,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )

    # Local contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    # Gentle sharpening
    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0,
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.4,
        blurred,
        -0.4,
        0,
    )

    return sharpened


def run_tesseract_layout(
    image: np.ndarray,
    psm: int = 3,
    language: str = "eng",
) -> dict[str, Any]:
    """
    OCR with word coordinates.

    The coordinates are essential for preserving the structure
    of Shipping Bill forms and tables.
    """

    pil_image = Image.fromarray(image)

    config = (
        f"--oem 3 --psm {psm} "
        "-c preserve_interword_spaces=1"
    )

    data = pytesseract.image_to_data(
        pil_image,
        lang=language,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    words: list[dict[str, Any]] = []
    confidences: list[float] = []

    total = len(data["text"])

    for i in range(total):

        text = data["text"][i].strip()

        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            confidence = -1

        if confidence < 0:
            continue

        word = {
            "text": text,
            "confidence": round(
                confidence / 100,
                4,
            ),
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "right": (
                int(data["left"][i])
                + int(data["width"][i])
            ),
            "bottom": (
                int(data["top"][i])
                + int(data["height"][i])
            ),
            "block_num": int(
                data["block_num"][i]
            ),
            "par_num": int(
                data["par_num"][i]
            ),
            "line_num": int(
                data["line_num"][i]
            ),
        }

        words.append(word)
        confidences.append(confidence)

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0
    )

    layout_text = reconstruct_layout(words)

    return {
        "psm": psm,
        "text": layout_text,
        "words": words,
        "confidence": round(
            average_confidence / 100,
            4,
        ),
    }


def reconstruct_layout(
    words: list[dict[str, Any]],
) -> str:
    """
    Reconstruct OCR output using Tesseract's block/line
    information and word coordinates.

    This is deliberately NOT a simple:
        " ".join(words)

    because that destroys columns.
    """

    if not words:
        return ""

    # ------------------------------------------
    # Group by OCR block
    # ------------------------------------------

    blocks: dict[int, list[dict[str, Any]]] = {}

    for word in words:

        block_id = word["block_num"]

        blocks.setdefault(
            block_id,
            [],
        ).append(word)

    # ------------------------------------------
    # Sort blocks from top -> bottom,
    # then left -> right
    # ------------------------------------------

    ordered_blocks = sorted(
        blocks.values(),
        key=lambda block: (
            min(word["top"] for word in block),
            min(word["left"] for word in block),
        ),
    )

    output: list[str] = []

    for block_index, block in enumerate(
        ordered_blocks,
        start=1,
    ):

        output.append(
            f"[BLOCK {block_index}]"
        )

        # --------------------------------------
        # Group block into lines
        # --------------------------------------

        lines: dict[
            tuple[int, int],
            list[dict[str, Any]]
        ] = {}

        for word in block:

            key = (
                word["par_num"],
                word["line_num"],
            )

            lines.setdefault(
                key,
                [],
            ).append(word)

        ordered_lines = sorted(
            lines.values(),
            key=lambda line: (
                min(word["top"] for word in line),
                min(word["left"] for word in line),
            ),
        )

        for line in ordered_lines:

            line = sorted(
                line,
                key=lambda word: word["left"],
            )

            line_text = ""

            previous_right = None

            for word in line:

                left = word["left"]

                if previous_right is None:

                    # Preserve left indentation
                    indentation = max(
                        0,
                        int(left / 20),
                    )

                    line_text += (
                        " " * min(
                            indentation,
                            30,
                        )
                    )

                else:

                    gap = left - previous_right

                    # Convert physical pixel gap into spaces
                    if gap < 10:
                        spaces = 1
                    elif gap < 30:
                        spaces = 2
                    elif gap < 60:
                        spaces = 4
                    elif gap < 100:
                        spaces = 8
                    else:
                        spaces = 12

                    line_text += (
                        " " * spaces
                    )

                line_text += word["text"]

                previous_right = word["right"]

            output.append(
                line_text.rstrip()
            )

        output.append("")

    return "\n".join(output).strip()


def ocr_image(
    image: Image.Image,
    language: str = "eng",
) -> dict[str, Any]:
    """
    Run multiple page-segmentation modes.

    PSM 3:
        automatic page layout

    PSM 11:
        sparse text

    We keep both results because the extractor can later
    use the word coordinates rather than relying on one
    flattened text result.
    """

    processed = prepare_image(
        image
    )

    variants = []

    for psm in [3, 11]:

        result = run_tesseract_layout(
            processed,
            psm=psm,
            language=language,
        )

        variants.append(result)

    # Prefer PSM 3 as the primary representation.
    primary = next(
        (
            item
            for item in variants
            if item["psm"] == 3
        ),
        variants[0],
    )

    return {
        "text": primary["text"],
        "confidence": primary["confidence"],
        "words": primary["words"],
        "variants": variants,
    }


def pdf_page_to_image(
    page: pymupdf.Page,
    dpi: int = 300,
) -> Image.Image:

    pixmap = page.get_pixmap(
        dpi=dpi,
        colorspace=pymupdf.csRGB,
        alpha=False,
    )

    return Image.open(
        io.BytesIO(
            pixmap.tobytes("png")
        )
    )


def extract_pdf_text(
    pdf_path: Path,
    language: str = "eng",
) -> dict[str, Any]:

    document = pymupdf.open(
        pdf_path
    )

    pages = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):

        native_text = page.get_text(
            "text",
            sort=True,
        ).strip()

        if len(native_text) >= 30:

            pages.append(
                {
                    "page": page_number,
                    "method": "native_text",
                    "text": native_text,
                    "confidence": 1.0,
                }
            )

        else:

            image = pdf_page_to_image(
                page,
                dpi=300,
            )

            result = ocr_image(
                image,
                language=language,
            )

            pages.append(
                {
                    "page": page_number,
                    "method": "ocr",
                    "text": result["text"],
                    "confidence": result["confidence"],
                    "words": result["words"],
                    "variants": result["variants"],
                }
            )

    document.close()

    combined_text = "\n\n".join(
        f"--- PAGE {page['page']} ---\n"
        f"{page['text']}"
        for page in pages
    )

    return {
        "file": str(pdf_path),
        "type": "pdf",
        "text": combined_text,
        "pages": pages,
    }


def extract_image_text(
    image_path: Path,
    language: str = "eng",
) -> dict[str, Any]:

    image = Image.open(
        image_path
    )

    result = ocr_image(
        image,
        language=language,
    )

    return {
        "file": str(image_path),
        "type": "image",
        "text": result["text"],
        "confidence": result["confidence"],
        "words": result["words"],
        "variants": result["variants"],
    }


def extract_text(
    file_path: str | Path,
    language: str = "eng",
) -> dict[str, Any]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if extension == ".pdf":

        return extract_pdf_text(
            path,
            language=language,
        )

    return extract_image_text(
        path,
        language=language,
    )