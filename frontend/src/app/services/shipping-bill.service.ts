import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ExtractionResult {
  port_code: string | null;
  shipping_bill_number: string | null;
  shipping_date: string | null;
  needs_review?: boolean;
}

export interface SubmitResult {
  success: boolean;
  database_id: number;
  port_code: string;
  shipping_bill_number: string;
  shipping_date: string | null;
}

@Injectable({
  providedIn: 'root',
})
export class ShippingBillService {

  private readonly http = inject(HttpClient);

  private readonly apiUrl =
    'http://127.0.0.1:8000';

  extract(
    file: File,
  ): Observable<ExtractionResult> {

    const formData = new FormData();

    formData.append(
      'file',
      file,
    );

    return this.http.post<ExtractionResult>(
      `${this.apiUrl}/extract`,
      formData,
    );
  }

  submit(
    data: {
      port_code: string;
      shipping_bill_number: string;
      shipping_date: string | null;
    },
  ): Observable<SubmitResult> {

    return this.http.post<SubmitResult>(
      `${this.apiUrl}/submit`,
      data,
    );
  }
}