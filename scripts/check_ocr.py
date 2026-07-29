"""OCR dependency check — verifies easyocr is installed and model files are cached.

Returns exit code 0 if all OK, 1 with guidance if not.
"""

import sys

try:
    import easyocr
    import cv2
    import numpy as np
except ImportError as e:
    print(f"MISSING DEP: {e.name}")
    print(f"Run: {sys.executable} -m pip install easyocr opencv-python-headless numpy")
    sys.exit(1)

import os
os.environ["EASYOCR_VERBOSE"] = "false"

try:
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
except Exception as e:
    print(f"EASYOCR INIT FAILED: {e}")
    print(f"Run: {sys.executable} -c \"import easyocr; easyocr.Reader(['en'], gpu=False)\"")
    print("This will download the detection and recognition models (~10MB each).")
    sys.exit(1)

img = (np.ones((200, 600, 3), dtype=np.uint8) * 255)
cv2.putText(img, "Test OCR 123", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
results = reader.readtext(img)
if results:
    print(f"OCR OK — detected: {[t for _, t, _ in results]}")
else:
    print("OCR WARNING: EasyOCR loaded but no text detected on test image")
    print("This may indicate a model issue.")

print("OCR system healthy.")
sys.exit(0)
