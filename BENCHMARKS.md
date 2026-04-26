# Pro Tier Benchmark Harness

This harness measures extraction quality against your real labeled documents.

## Folder layout

- `data/benchmarks/docs/` contains PDFs or images to evaluate.
- `data/benchmarks/labels/` contains one JSON label file per document.
- `data/benchmarks/reports/` receives generated benchmark reports.

## Label file format

```json
{
  "file": "invoice_001.pdf",
  "expected": {
    "classification": "invoice",
    "fields": {
      "invoice_number": "INV-2026-001",
      "invoice_date": "2026-03-10",
      "total_amount": "1299.99",
      "currency": "EUR"
    }
  }
}
```

## Run

From repository root:

```powershell
python scripts/benchmark_harness.py
```

Or with custom folders:

```powershell
python scripts/benchmark_harness.py --docs data/benchmarks/docs --labels data/benchmarks/labels --reports data/benchmarks/reports
```

## Output

The script writes a JSON report to `data/benchmarks/reports/benchmark-YYYYMMDD-HHMMSS.json` with:

- `classification_accuracy`
- `field_accuracy`
- per-document error details
- per-field match details
