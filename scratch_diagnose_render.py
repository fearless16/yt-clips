import cv2
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

from face_os.pipeline import FaceOSPipeline
from face_os.physical_renderer import LightingModel

def main():
    p = FaceOSPipeline()
    p.enroll()
    
    cap = cv2.VideoCapture("input/video.mp4")
    # Read up to frame 4
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Maintain tracker state on all frames
        face_track = p.tracker.process_frame(frame, i)
        landmarks = None
        if face_track and face_track.smooth_bbox is not None and face_track.mesh_478 is not None:
            from face_os import landmarks as lm_module
            landmarks = lm_module.extract_landmarks(frame, face_track.mesh_478)
        
        if i == 4:
            # We are at frame 4!
            print("=== Frame 4 Analysis ===")
            print(f"face_track: {face_track is not None}, landmarks: {landmarks is not None}")
            if face_track:
                print(f"  smooth_bbox: {face_track.smooth_bbox}")
                print(f"  mesh_478: {face_track.mesh_478 is not None}")
            
            crop_plan = p.crop.plan_crop(frame.shape[:2], face_track, landmarks)
            from face_os import crop_planner
            cropped = crop_planner.apply_crop(frame, crop_plan)
            
            print(f"Original crop: mean={np.mean(cropped):.3f}, std={np.std(cropped):.3f}")
            
            # Step 2: Convert to RGB float
            source_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Step 3: Decompose
            source_decomposer = p.identity_state._intrinsic_decomposer
            source_intrinsic = source_decomposer.decompose(source_rgb)
            
            print(f"Decomposition:")
            print(f"  albedo: mean={np.mean(source_intrinsic.albedo):.3f}, std={np.std(source_intrinsic.albedo):.3f}")
            print(f"  shading: mean={np.mean(source_intrinsic.shading):.3f}, std={np.std(source_intrinsic.shading):.3f}")
            print(f"  specular: mean={np.mean(source_intrinsic.specular):.3f}, std={np.std(source_intrinsic.specular):.3f}")
            
            # Step 4: Estimate lighting
            shading_mean = float(np.mean(source_intrinsic.shading))
            lighting = LightingModel(
                ambient=shading_mean * 0.3,
                diffuse_intensity=shading_mean * 0.8,
            )
            print(f"Estimated Lighting:")
            print(f"  ambient={lighting.ambient:.3f}, diffuse_intensity={lighting.diffuse_intensity:.3f}")
            
            # Step 5: Render
            from face_os.dense_geometry import DenseGeometryEstimator
            dense_estimator = DenseGeometryEstimator()
            dense_geometry = dense_estimator.estimate(landmarks.points[:, :2])
            
            rendered_output = p._face_renderer._renderer.render_with_mesh(
                albedo=source_intrinsic.albedo,
                mesh_vertices=dense_geometry.vertices,
                mesh_faces=dense_geometry.faces,
                shading=source_intrinsic.shading,
                lighting=lighting,
                image_size=source_intrinsic.albedo.shape[:2],
            )
            
            print(f"Rendered Output:")
            print(f"  rendered: mean={np.mean(rendered_output.rendered):.3f}, std={np.std(rendered_output.rendered):.3f}")
            print(f"  diffuse: mean={np.mean(rendered_output.diffuse_component):.3f}")
            print(f"  specular: mean={np.mean(rendered_output.specular_component):.3f}")
            print(f"  ambient: mean={np.mean(rendered_output.ambient_component):.3f}")
            print(f"  detail: mean={np.mean(rendered_output.detail_component):.3f}")
            
            # Step 6: Post-process & Detail residual
            rendered_face = np.clip(rendered_output.rendered, 0.0, 1.0).astype(np.float32)
            rendered_face_u8 = (rendered_face * 255.0).astype(np.uint8)
            detailed = p._inject_detail_residual(
                rendered_face_u8,
                source_intrinsic,
                face_mask=None,
                strength=0.25,
            )
            print(f"Detail residual injection:")
            print(f"  detailed: mean={np.mean(detailed):.3f}")
            
    cap.release()

if __name__ == "__main__":
    main()
