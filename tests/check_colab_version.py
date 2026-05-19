"""Check remote ref_grade.py version on Colab by comparing line count + keys."""
import sys
sys.path.insert(0, '.')
import ref_grade
p = ref_grade.enroll("expectation.png")
print(f"params: {len(p)}")
print(f"has_lut_a: {'_lut_a' in p}")
print(f"has_split_lut: {'_split_lut' in p}")
print(f"has_vignette_cache: hasattr(ref_grade, '_vignette_cache')={hasattr(ref_grade, '_vignette_cache')}")
