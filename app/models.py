diff --git a/app/models.py b/app/models.py
index e2c2211..5b4be97 100644
--- a/app/models.py
+++ b/app/models.py
@@ -1,4 +1,4 @@
-from typing import Literal, Optional, Dict, Any
+from typing import Literal, Optional, Dict, Any
 from pydantic import BaseModel, Field
 
 PatternName = Literal["crown_shelf_right_spike"]
@@ -24,6 +24,7 @@ class BestWindow(BaseModel):
     start_idx: int
     end_idx: int
     length_days: int
+    best_age_days: int
     candidate_windows_count: int
 
 
 class DebugSymbolInfo(BaseModel):
+    input_symbol: str | None = None
     resolved: bool = False
     coingecko_id: str | None = None
     status: str = "unknown"
@@ -34,6 +35,12 @@ class DebugSymbolInfo(BaseModel):
     http_status: int | None = None
     request_params: Dict[str, Any] | None = None
     error_message: str | None = None
+    auth_mode: str | None = None
+    base_url: str | None = None
+    api_key_present: bool | None = None
+    auth_header_name: str | None = None
+    candidate_windows_count: int | None = None
+    best_window: Dict[str, Any] | None = None
+    raw_similarity: float | None = None
+    label: str | None = None
 
 
 class ScanResult(BaseModel):
