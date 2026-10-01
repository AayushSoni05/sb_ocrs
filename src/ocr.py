from __future__ import annotations

from pathlib import Path
from typing import Any

import platform
import shutil

import cv2
import numpy as np
import pytesseract


# =========================================================
# TESSERACT PATH
# =========================================================
#
# On Windows, pytesseract can't find tesseract.exe on PATH by default,
# so point it at the common install location. On Mac/Linux, tesseract
# is normally already on PATH after `brew install tesseract` or
# `apt install tesseract-ocr`, so we leave pytesseract's default alone.

if platform.system() == "Windows":

    default_windows_path = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if Path(default_windows_path).exists():

        pytesseract.pytesseract.tesseract_cmd = default_windows_path


class ShippingBillOCR:

    def __init__(self) -> None:

        tesseract_cmd = pytesseract.pytesseract.tesseract_cmd

        found = (
            Path(tesseract_cmd).exists()
            if platform.system() == "Windows"
            else shutil.which(tesseract_cmd) is not None
        )

        if not found:

            raise FileNotFoundError(
                "Tesseract was not found.\n\n"
                "Install it first:\n"
                "  macOS:   brew install tesseract\n"
                "  Linux:   sudo apt install tesseract-ocr\n"
                "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
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
        # Shipping Bill header:
        #
        # PORT CODE | SB NO | SB DATE
        #
        # We include the complete top-right information box
        # so the value row is available to split_header().
        # -------------------------------------------------

        x1 = int(width * 0.49)
        y1 = int(height * 0.038)

        x2 = int(width * 0.83)
        y2 = int(height * 0.135)

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
            scale=12,
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

    # Below this real (pre-upscale) pixel height, individual
    # characters are too few pixels tall for Tesseract to
    # read reliably - upscaling a crop this small just blurs
    # existing pixels, it can't add real detail. Flag it
    # instead of returning a confident-looking guess.
    MIN_READABLE_HEIGHT_PX = 20

    @staticmethod
    def ocr_cell(
        cell: np.ndarray,
        whitelist: str,
    ) -> list[dict[str, Any]]:

        results = []

        real_height = cell.shape[0]

        if real_height < ShippingBillOCR.MIN_READABLE_HEIGHT_PX:

            results.append(
                {
                    "variant": 0,
                    "text": None,
                    "confidence": 0.0,
                    "low_resolution": True,
                    "note": (
                        f"Cropped cell is only {real_height}px tall "
                        f"(need >= {ShippingBillOCR.MIN_READABLE_HEIGHT_PX}px). "
                        "Source image resolution is too low for reliable "
                        "OCR here - use a higher-DPI scan/photo."
                    ),
                }
            )

            return results

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
    # OCR THE WHOLE VALUE ROW (fallback for low-res images)
    # =====================================================
    #
    # Splitting into 3 tiny cells works well on high-res
    # scans, but on a low-res image each cell can be only a
    # few pixels tall - too little real detail for Tesseract,
    # even after upscaling (upscaling blurs, it doesn't add
    # information). Reading the WHOLE row as one line gives
    # Tesseract more surrounding pixels/context per character,
    # which tends to hold up much better at low resolution.

    @staticmethod
    def ocr_whole_value_row(
        value_row: np.ndarray,
    ) -> list[dict[str, Any]]:

        results = []

        variants = (
            ShippingBillOCR.preprocessing_variants(
                value_row
            )
        )

        whitelist = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/ "
        )

        for variant_number, variant in enumerate(
            variants,
            start=1,
        ):

            # psm 6 = "assume a single uniform block of
            # text" - better suited to a whole row of
            # multiple values than psm 7 (single line,
            # meant for one tight value).

            config = (
                "--oem 3 "
                "--psm 6 "
                "-c preserve_interword_spaces=1 "
                f"-c tessedit_char_whitelist={whitelist}"
            )

            text = pytesseract.image_to_string(
                variant,
                lang="eng",
                config=config,
            ).strip()

            if not text:
                continue

            results.append(
                {
                    "variant": variant_number,
                    "text": text,
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

        # -------------------------------------------------
        # The header crop contains:
        #
        #     PORT CODE | SB NO | SB DATE
        #     INKKU6    | 3705955 | 30-MAY-26
        #
        # followed by:
        #
        # IEC/Br
        # GSTIN/TYPE
        # CB CODE
        # ...
        #
        # We only want the first VALUE row.
        # -------------------------------------------------

        value_y1 = int(
            height * 0.12
        )

        value_y2 = int(
            height * 0.30
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
        #
        # PORT CODE
        #   ~0% -> 32%
        #
        # SB NO
        #   ~32% -> 66%
        #
        # SB DATE
        #   ~66% -> 100%
        # -------------------------------------------------

        port_x1 = 0

        port_x2 = int(
            row_width * 0.32
        )

        sb_x1 = int(
            row_width * 0.32
        )

        sb_x2 = int(
            row_width * 0.66
        )

        date_x1 = int(
            row_width * 0.66
        )

        date_x2 = row_width

        return {
            "value_row": value_row,

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
        # Split into cells
        # -------------------------------------------------

        cells = self.split_header(
            header
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

        # -------------------------------------------------
        # Fallback: OCR the whole (unsplit) value row too.
        # More context per character than the tiny individual
        # cells above - helps a lot on low-resolution images.
        # -------------------------------------------------

        whole_row_results = self.ocr_whole_value_row(
            cells["value_row"]
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
            "whole_row_candidates": whole_row_results,
        }