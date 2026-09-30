from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract


# =========================================================
# TESSERACT PATH
# =========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


class ShippingBillOCR:

    def __init__(self) -> None:

        if not Path(
            pytesseract.pytesseract.tesseract_cmd
        ).exists():

            raise FileNotFoundError(
                "Tesseract was not found at:\n"
                f"{pytesseract.pytesseract.tesseract_cmd}\n\n"
                "Update the path in src/ocr.py."
            )

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    @staticmethod
    def load_image(
        file_path: str | Path,
    ) -> np.ndarray:

        file_path = Path(
            file_path
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        image = cv2.imread(
            str(file_path)
        )

        if image is None:

            raise ValueError(
                f"Unable to read image: {file_path}"
            )

        return image

    # =====================================================
    # CROP HEADER TABLE
    # =====================================================

    @staticmethod
    def crop_header(
        image: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:

        height, width = image.shape[:2]

        # -------------------------------------------------
        # Coordinates determined from the supplied
        # Shipping Bill layout.
        #
        # The header table is near the upper-right.
        # -------------------------------------------------

        x1 = int(width * 0.49)
        y1 = int(height * 0.035)

        x2 = int(width * 0.83)
        y2 = int(height * 0.145)

        crop = image[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:

            raise ValueError(
                "Header crop is empty."
            )

        return (
            crop,
            (
                x1,
                y1,
                x2,
                y2,
            ),
        )

    # =====================================================
    # UPSCALE
    # =====================================================

    @staticmethod
    def upscale(
        image: np.ndarray,
        scale: int = 8,
    ) -> np.ndarray:

        return cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    # =====================================================
    # PREPROCESS VARIANTS
    # =====================================================

    @staticmethod
    def preprocessing_variants(
        image: np.ndarray,
    ) -> list[np.ndarray]:

        enlarged = ShippingBillOCR.upscale(
            image,
            scale=8,
        )

        gray = cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2GRAY,
        )

        # -------------------------------------------------
        # Variant 1: grayscale
        # -------------------------------------------------

        variant_gray = gray

        # -------------------------------------------------
        # Variant 2: CLAHE
        # -------------------------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        variant_clahe = clahe.apply(
            gray
        )

        # -------------------------------------------------
        # Variant 3: OTSU
        # -------------------------------------------------

        _, variant_otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU,
        )

        # -------------------------------------------------
        # Variant 4: adaptive threshold
        # -------------------------------------------------

        variant_adaptive = (
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
        )

        return [
            variant_gray,
            variant_clahe,
            variant_otsu,
            variant_adaptive,
        ]

    # =====================================================
    # OCR ONE CELL
    # =====================================================

    @staticmethod
    def ocr_cell(
        cell: np.ndarray,
        whitelist: str,
    ) -> list[dict[str, Any]]:

        results = []

        variants = (
            ShippingBillOCR.preprocessing_variants(
                cell
            )
        )

        for variant_number, variant in enumerate(
            variants,
            start=1,
        ):

            config = (
                "--oem 3 "
                "--psm 7 "
                "-c preserve_interword_spaces=1 "
                f"-c tessedit_char_whitelist={whitelist}"
            )

            data = pytesseract.image_to_data(
                variant,
                lang="eng",
                config=config,
                output_type=pytesseract.Output.DICT,
            )

            texts = []

            confidences = []

            for i, raw_text in enumerate(
                data["text"]
            ):

                text = raw_text.strip()

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

                texts.append(
                    text
                )

                confidences.append(
                    confidence / 100
                )

            if not texts:
                continue

            combined = "".join(
                texts
            ).strip()

            average_confidence = (
                sum(confidences)
                / len(confidences)
                if confidences
                else 0.0
            )

            results.append(
                {
                    "variant": variant_number,
                    "text": combined,
                    "confidence": round(
                        average_confidence,
                        4,
                    ),
                }
            )

        return results

    # =====================================================
    # SPLIT HEADER INTO THREE CELLS
    # =====================================================

    @staticmethod
    def split_header(
        header: np.ndarray,
    ) -> dict[str, np.ndarray]:

        height, width = (
            header.shape[:2]
        )

        # The first data row is immediately below
        # the green headings.
        #
        # We intentionally crop the VALUE row only.

        value_y1 = int(
            height * 0.27
        )

        value_y2 = int(
            height * 0.52
        )

        value_row = header[
            value_y1:value_y2,
            :,
        ]

        row_height, row_width = (
            value_row.shape[:2]
        )

        # -------------------------------------------------
        # Three columns
        # -------------------------------------------------

        # Based on supplied document:
        #
        # Port Code : ~28%
        # SB No     : ~61%
        # SB Date   : remaining
        #
        # Small overlap avoids cutting characters.
        # -------------------------------------------------

        port_x1 = 0
        port_x2 = int(
            row_width * 0.30
        )

        sb_x1 = int(
            row_width * 0.27
        )

        sb_x2 = int(
            row_width * 0.62
        )

        date_x1 = int(
            row_width * 0.59
        )

        date_x2 = row_width

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

    # =====================================================
    # COMPLETE EXTRACTION
    # =====================================================

    def extract(
        self,
        file_path: str | Path,
    ) -> dict[str, Any]:

        image = self.load_image(
            file_path
        )

        header, coordinates = (
            self.crop_header(
                image
            )
        )

        # -------------------------------------------------
        # Save debugging image
        # -------------------------------------------------

        Path(
            "output"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            "output/header_crop.png",
            header,
        )

        # -------------------------------------------------
        # Split into cells
        # -------------------------------------------------

        cells = self.split_header(
            header
        )

        # Save each cell
        cv2.imwrite(
            "output/port_cell.png",
            cells["port_code"],
        )

        cv2.imwrite(
            "output/shipping_bill_cell.png",
            cells[
                "shipping_bill_number"
            ],
        )

        cv2.imwrite(
            "output/date_cell.png",
            cells["shipping_date"],
        )

        # -------------------------------------------------
        # OCR each field independently
        # -------------------------------------------------

        port_results = self.ocr_cell(
            cells["port_code"],
            whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )

        sb_results = self.ocr_cell(
            cells[
                "shipping_bill_number"
            ],
            whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-",
        )

        date_results = self.ocr_cell(
            cells["shipping_date"],
            whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/:.",
        )

        return {
            "header_coordinates": {
                "x1": coordinates[0],
                "y1": coordinates[1],
                "x2": coordinates[2],
                "y2": coordinates[3],
            },

            "port_code_candidates": port_results,

            "shipping_bill_candidates": sb_results,

            "shipping_date_candidates": date_results,
        }