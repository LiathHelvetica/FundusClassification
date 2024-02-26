from pandas import read_csv, concat, Series

from constants import VALIDATION_LABELS_PATH, TEST_LABELS_PATH, TRAIN_LABELS_PATH, ALL_LABEL_PATH
from utils import lthv_read_csv

df = read_csv(ALL_LABEL_PATH)
disease_counts = df['Disease'].value_counts()
print(disease_counts)
print(f"n_classes: {len(disease_counts)}")
