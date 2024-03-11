import cv2
import numpy as np
import os
import albumentations as al

from constants import ALL_DATA_PATH, ALL_CROPPED_DATA_PATH, ALL_PATH

TO_SIZE = 1388

IN_PATH = f"{ALL_PATH}/Data"
OUT_PATH = f"{ALL_PATH}/Redo"
REDO = [
  "test516.png"
]

"""
    blurred = cv2.blur(img, (3, 3))
    canny = cv2.Canny(blurred, 50, 200)

    pts = np.argwhere(canny > 0)
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)

    cropped = img[y1:y2, x1:x2]
"""

"""
tr = al.Crop(292, 31, 1759, 1501)
cropped = tr(image=img)["image"]
"""


for f_name in REDO:
  try:
    img = cv2.imread(f"{IN_PATH}/{f_name}")


    blurred = cv2.blur(img, (3, 3))
    canny = cv2.Canny(blurred, 50, 120)

    pts = np.argwhere(canny > 0)
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)

    cropped = img[y1:y2, x1:x2]

    cv2.imwrite(f"{OUT_PATH}/{f_name}", cropped)
  except Exception:
    print(f"Failed: {f_name}")
