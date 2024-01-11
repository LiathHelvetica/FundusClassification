from augmentation_pipelines_old import read_imgs_group, list_paths_to_imgs, apply_transformation_to_batch
from constants import TEST_OUT_PATH, OUT_PATH, TRAIN_OUT_PATH, VALIDATION_OUT_PATH
import albumentations as al

SIZE = 224
IN = VALIDATION_OUT_PATH
OUT = f"{OUT_PATH}/{SIZE}/valid"

if __name__ == "__main__":
  f_groups = read_imgs_group(IN, group_size=10)
  for group in f_groups:
    batch = list_paths_to_imgs(group, IN, batch_name="")

    batch = apply_transformation_to_batch(al.Resize(SIZE, SIZE, p=1.0), batch, t_name="")

    for im in batch:
      im.save_to(OUT)
