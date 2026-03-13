"""
Example integration file.

Copy-paste this into your existing FastAPI app if you want a minimal working wire-up.
Replace `from your_project.scanner_service import scan_service` with your real import.
"""

from fastapi import FastAPI
from pattern_scanner_patch.router import router, run_scan_service, ScanRequest

app = FastAPI()
app.include_router(router)

# Example monkey-patch for the service function.
# Replace this whole block with your real implementation.
def _real_run_scan_service(req: ScanRequest):
    from your_project.scanner_service import scan_service  # change this import
    return scan_service.scan(req.model_dump())

import pattern_scanner_patch.router as patched_router
patched_router.run_scan_service = _real_run_scan_service
