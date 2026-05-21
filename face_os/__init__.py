"""
face_os — Personal Face Operating System.

A face-aware vertical video pipeline that treats identity as a persistent
appearance field, not a per-frame rendering problem.

Core philosophy:
  - Overfit is the feature. One face, one environment, one camera.
  - Face is a dynamic appearance function A(u,v,θ,L,t), not a static mesh.
  - Pixels are noisy photon observations — accumulate confidence over time.
  - Identity inertia: ΔI_identity ≪ ΔI_source.
  - Eyes dominate perception — always render eyes at highest fidelity.

Modules:
  1. ingest            — Video ingestion + audio sync
  2. detect_track      — Face detection + temporal tracking
  3. landmarks         — 478-point landmarks (MediaPipe) + head pose estimation
  4. canonical_map     — Per-identity UV atlas + appearance field
  5. crop_planner      — 9:16 crop with face-aware headroom
  6. temporal_stabilize — Identity inertia + flicker suppression
  7. face_enhance      — Eye-dominant rendering + skin refinement
  8. identity_memory   — Photic memory + confidence accumulation
  9. compositor        — Confidence-weighted per-pixel compositing
  10. export_qc        — Final encode + quality checks
"""

__version__ = "0.1.0"
