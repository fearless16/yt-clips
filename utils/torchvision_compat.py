"""
torchvision_compat.py — Patch missing torchvision.transforms.functional_tensor.

torchvision >= 0.17 removed `functional_tensor` but basicsr/realesrgan still
import it.  This module patches a compatible shim so downstream imports succeed.

Import this BEFORE any realesrgan/basicsr import:
    import utils.torchvision_compat  # noqa: F401
"""
import sys
import types
import torchvision.transforms.functional as _F

if "torchvision.transforms.functional_tensor" not in sys.modules:
    _ft = types.ModuleType("torchvision.transforms.functional_tensor")
    _ft.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _ft
