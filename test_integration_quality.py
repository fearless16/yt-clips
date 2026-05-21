import cv2
import numpy as np
from export import export_clip_integrated
from ref_grade import enroll
from pathlib import Path

def calculate_lab_distance(img1, img2):
    # Ensure same size
    h, w = img1.shape[:2]
    img2_res = cv2.resize(img2, (w, h))
    
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2_res, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    dist = np.sqrt(np.sum((lab1 - lab2)**2, axis=2))
    return np.mean(dist)

def main():
    source = "clips_test/test_clip.mp4"
    reference = "expectation.png"
    output = "output/integration_test_result.mp4"
    Path("output").mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {source}...")
    # Process first 5 seconds
    res = export_clip_integrated(
        video_path=source,
        start=0.0,
        end=5.0,
        output_path=output,
        clip_id="quality_test"
    )
    
    if not res:
        print("Export failed")
        return

    # Extract a frame from the middle of the result
    cap = cv2.VideoCapture(output)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30) # frame 30
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not extract frame")
        return
    
    # Load reference
    ref_img = cv2.imread(reference)
    
    # Basic Analysis
    # 1. LAB Distance (Overall)
    dist = calculate_lab_distance(frame, ref_img)
    
    # 2. Face brightness check
    # We assume the face is roughly in the center-top
    h, w = frame.shape[:2]
    face_region = frame[int(h*0.2):int(h*0.4), int(w*0.3):int(w*0.7)]
    face_l = np.mean(cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)[:, :, 0])
    
    ref_face_region = ref_img[int(ref_img.shape[0]*0.2):int(ref_img.shape[0]*0.4), 
                              int(ref_img.shape[1]*0.3):int(ref_img.shape[1]*0.7)]
    ref_l = np.mean(cv2.cvtColor(ref_face_region, cv2.COLOR_BGR2LAB)[:, :, 0])
    
    print("\n--- QUALITY RESULTS ---")
    print(f"Overall LAB Distance: {dist:.2f}")
    print(f"Result Face L: {face_l:.2f}")
    print(f"Reference Face L: {ref_l:.2f}")
    print(f"L Delta: {abs(face_l - ref_l):.2f}")

if __name__ == "__main__":
    main()
