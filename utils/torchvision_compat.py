"""
torchvision_compat.py — Patch missing torchvision.transforms.functional_tensor
and PIL._util.is_directory for basicsr/realesrgan/gfpgan compatibility.

Import this BEFORE any realesrgan/basicsr import:
    import utils.torchvision_compat  # noqa: F401
"""
import os
import sys
import types

# Patch PIL._util.is_directory (removed in Pillow 10+)
try:
    import PIL._util
    if not hasattr(PIL._util, "is_directory"):
        PIL._util.is_directory = os.path.isdir
except ImportError:
    pass

# Patch torchvision.transforms.functional_tensor (removed in torchvision 0.17+)
import torchvision.transforms.functional as _F

if "torchvision.transforms.functional_tensor" not in sys.modules:
    _ft = types.ModuleType("torchvision.transforms.functional_tensor")
    _ft.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _ft
