import {
  Component,
  ChangeDetectorRef,
  inject,
} from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  ShippingBillService,
  ExtractionResult,
} from './services/shipping-bill.service';
import { finalize } from 'rxjs';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    FormsModule,
  ],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {

  private readonly shippingBillService =
    inject(ShippingBillService);

  private readonly changeDetectorRef =
    inject(ChangeDetectorRef);

  selectedFile: File | null = null;

  portCode = '';

  shippingBillNumber = '';

  shippingDate: string | null = null;

  isExtracting = false;

  isSubmitting = false;

  message = '';

  errorMessage = '';


  onFileSelected(
    event: Event,
  ): void {

    const input =
      event.target as HTMLInputElement;

    const file =
      input.files?.[0];

    if (!file) {
      return;
    }

    this.selectedFile = file;

    this.portCode = '';

    this.shippingBillNumber = '';

    this.shippingDate = null;

    this.message = '';

    this.errorMessage = '';

    this.isExtracting = true;


    this.shippingBillService
      .extract(file)
      .subscribe({

        next: (
          result: ExtractionResult,
        ) => {

          this.isExtracting = false;

          this.portCode =
            result.port_code ?? '';

          this.shippingBillNumber =
            result.shipping_bill_number ?? '';

          this.shippingDate =
            result.shipping_date ?? '';

          if (
            result.needs_review
            || !this.portCode
            || !this.shippingBillNumber
          ) {

            this.errorMessage =
              'The document could not be reliably extracted.';

          } else {

            this.message =
              'Fields extracted successfully.';
          }
          this.changeDetectorRef.markForCheck();
        },

        error: (
          error,
        ) => {

          this.isExtracting = false;

          this.errorMessage =
            error?.error?.detail
            ?? 'Extraction failed.';
            this.changeDetectorRef.markForCheck();
        },
      });
  }


  submitForm(): void {

  this.message = '';

  this.errorMessage = '';

  if (
    !this.portCode
    || !this.shippingBillNumber
  ) {

    this.errorMessage =
      'Port Code and Shipping Bill Number are required.';

    return;
  }

  this.isSubmitting = true;

  this.shippingBillService
    .submit({
      port_code: this.portCode,
      shipping_bill_number:
        this.shippingBillNumber,
      shipping_date:
        this.shippingDate,
    })
    .pipe(
      finalize(() => {
        this.isSubmitting = false;
        this.changeDetectorRef.markForCheck();
      })
    )
    .subscribe({

      next: (result) => {

        this.message =
          `Saved successfully. Database ID: ${result.database_id}`;

        this.errorMessage = '';
        this.changeDetectorRef.markForCheck();

      },

      error: (error) => {

        this.errorMessage =
          error?.error?.detail
          ?? 'Database submission failed.';

        this.message = '';
        this.changeDetectorRef.markForCheck();

      },

    });
  }
}