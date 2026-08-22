# Transcript Desk

Transcript Desk is a local, read-only side-by-side viewer for LOC scans and
CHURRO prediction JSONL.

From the repository root in PowerShell:

```powershell
.\tools\transcript_desk\OPEN_VIEWER.ps1 -Port 8815
```

To use a different prediction file:

```powershell
.\tools\transcript_desk\OPEN_VIEWER.ps1 -Port 8815 `
  -Predictions .\path\to\predictions.jsonl
```

Features:

- scan and transcript displayed side by side;
- automatic JSONL reload every 15 seconds;
- page-ID and transcript search;
- complete, incomplete, truncated, omission-marker, and LOC-review-flag filters;
- LOC-format/audit badges when viewing `loc_drafts.jsonl`;
- keyboard navigation and image zoom;
- copy or download one draft transcript.

The viewer never edits predictions and never treats OCR output as ground truth.
Python's standard library is sufficient for the viewer server.
