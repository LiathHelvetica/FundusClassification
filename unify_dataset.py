from pandas import read_csv, concat

from constants import VALIDATION_LABELS_PATH, TEST_LABELS_PATH, TRAIN_LABELS_PATH, HEALTHY_LABEL, ALL_LABEL_PATH


def get_column_name(row) -> str:
  for col in row.index:
    if row[col] == 1:
      return col
  return HEALTHY_LABEL


out = ALL_LABEL_PATH

df_val = read_csv(VALIDATION_LABELS_PATH)
df_test = read_csv(TEST_LABELS_PATH)
df_train = read_csv(TRAIN_LABELS_PATH)

df_val["OriginalDataset"] = "val"
df_test["OriginalDataset"] = "test"
df_train["OriginalDataset"] = "train"

df_all = concat([df_val, df_test, df_train])

df_all["Disease"] = df_all.drop(["ID", "Disease_Risk"], axis=1).apply(get_column_name, axis=1)
df_all = df_all[["ID", "OriginalDataset", "Disease"]]

df_all["ID"] = df_all.apply(lambda r: f"{r['OriginalDataset']}{r['ID']}", axis=1)
df_all = df_all[["ID", "Disease"]]

df_all.to_csv(out, index=False)

