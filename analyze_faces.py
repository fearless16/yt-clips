import cv2
import numpy as np

source = cv2.imread("source_frame.jpg")
if source is None:
    print("Cannot read source_frame.jpg")
    exit(1)

gray_source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

import glob
photos = glob.glob("photos/*.png") + glob.glob("photos/*.jpg")
print("Found reference photos:", photos)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in source frame
source_faces = face_cascade.detectMultiScale(gray_source, 1.05, 3)
print(f"Detected {len(source_faces)} faces in source.")

# We want to find which face matches the user.
# Let's use Histogram comparison or Structural Similarity (SSIM) or simply find the face in the bottom corners.
for (x, y, w, h) in source_faces:
    print(f"Face at x={x}, y={y}, w={w}, h={h}")

# Also analyze expectation.png
exp = cv2.imread("expectation.png")
if exp is not None:
    print("expectation shape:", exp.shape)
    exp_gray = cv2.cvtColor(exp, cv2.COLOR_BGR2GRAY)
    exp_faces = face_cascade.detectMultiScale(exp_gray, 1.05, 3)
    print("Faces in expectation:", exp_faces)
else:
    print("Cannot read expectation.png")

