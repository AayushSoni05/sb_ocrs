from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ocr import ShippingBillOCR
from src.extractor import extract_shipping_bill_fields
from src.database import save_shipping_bill
from pdf import pdf_page_to_png


app = FastAPI(
    title="Shipping Bill OCR API",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# RESPONSE / SUBMIT MODEL
# =========================================================

class ShippingBillSubmission(BaseModel):

    port_code: str

    shipping_bill_number: str

    shipping_date: str | None = None


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health() -> dict[str, str]:

    return {
        "status": "ok"
    }


# =========================================================
# EXTRACT PDF
# =========================================================

@app.post("/extract")
async def extract_shipping_bill(
    file: UploadFile = File(...),
):
    """
    Receive a PDF/image from Angular.

    OCR the document and return extracted JSON.

    IMPORTANT:
    This endpoint does NOT insert into PostgreSQL.
    """

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename supplied.",
        )

    extension = (
        Path(file.filename)
        .suffix
        .lower()
    )

    allowed_extensions = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Use PDF, PNG, JPG, JPEG, TIF or TIFF."
            ),
        )

    temporary_input = None
    temporary_render = None

    try:

        # -------------------------------------------------
        # Save uploaded file temporarily
        # -------------------------------------------------

        suffix = extension

        temp_input = tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        )

        temporary_input = Path(
            temp_input.name
        )

        contents = await file.read()

        temp_input.write(
            contents
        )

        temp_input.close()

        # -------------------------------------------------
        # PDF -> temporary 600 DPI PNG
        # -------------------------------------------------

        if extension == ".pdf":

            temp_render_file = (
                tempfile.NamedTemporaryFile(
                    suffix=".png",
                    delete=False,
                )
            )

            temporary_render = Path(
                temp_render_file.name
            )

            temp_render_file.close()

            image_path = pdf_page_to_png(
                temporary_input,
                output_path=temporary_render,
                page_number=0,
                dpi=600,
            )

        else:

            image_path = temporary_input

        # -------------------------------------------------
        # OCR
        # -------------------------------------------------

        ocr = ShippingBillOCR()

        ocr_result = ocr.extract(
            image_path
        )

        # -------------------------------------------------
        # STRUCTURED EXTRACTION
        # -------------------------------------------------

        fields = extract_shipping_bill_fields(
            ocr_result
        )

        return fields

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    finally:

        if temporary_render is not None:

            temporary_render.unlink(
                missing_ok=True
            )

        if temporary_input is not None:

            temporary_input.unlink(
                missing_ok=True
            )


# =========================================================
# SUBMIT TO DATABASE
# =========================================================

@app.post("/submit")
def submit_shipping_bill(
    submission: ShippingBillSubmission,
):

    if not submission.port_code:

        raise HTTPException(
            status_code=400,
            detail="Port code is required.",
        )

    if not submission.shipping_bill_number:

        raise HTTPException(
            status_code=400,
            detail="Shipping bill number is required.",
        )

    try:

        record_id = save_shipping_bill(
            {
                "port_code":
                    submission.port_code,

                "shipping_bill_number":
                    submission.shipping_bill_number,

                "shipping_date":
                    submission.shipping_date,
            }
        )

        return {
            "success": True,
            "database_id": record_id,
            "port_code":
                submission.port_code,
            "shipping_bill_number":
                submission.shipping_bill_number,
            "shipping_date":
                submission.shipping_date,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )