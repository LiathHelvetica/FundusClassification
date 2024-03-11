import os
import cv2
import albumentations as al
import tensorflow as tf

from constants import ALL_CROPPED_DATA_PATH, ALL_CROPPED_SQUARE_DATA_PATH, ALL_PATH

INPUT_PATH = f"{ALL_PATH}/Redo"
OUTPUT_PATH = f"{ALL_PATH}/Redo"

if __name__ == "__main__":
  for f_name in os.listdir(INPUT_PATH):
    img = cv2.imread(f"{INPUT_PATH}/{f_name}")

    size = max(img.shape[0], img.shape[1])

    out = tf.image.resize_with_pad(img, size, size).numpy()

    cv2.imwrite(f"{OUTPUT_PATH}/{f_name}", out)
