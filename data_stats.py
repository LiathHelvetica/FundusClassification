from pandas import read_csv, concat, Series

from constants import VALIDATION_LABELS_PATH, TEST_LABELS_PATH, TRAIN_LABELS_PATH


def lthv_read_csv(path, fid):
  df = read_csv(path)
  df.index = df.index.to_series().map(lambda i: f"{fid}-{i}")
  return df


df_val = lthv_read_csv(VALIDATION_LABELS_PATH, "val")
df_test = lthv_read_csv(TEST_LABELS_PATH, "tst")
df_train = lthv_read_csv(TRAIN_LABELS_PATH, "trn")

df_all = concat([df_val, df_test, df_train])

df_disease_count: Series = df_all.sum()
df_disease_count = df_disease_count[df_disease_count.index.map(lambda s: s not in {"ID"})]
print("Diseases count")
print(df_disease_count)

print("Healthy count")
print(df_all.shape[0] - df_disease_count["Disease_Risk"])

print("Can an eye have multiple diseases")
df_disease_sum = df_disease_count[df_disease_count.index.map(lambda s: s not in {"Disease_Risk"})].sum()
print(df_disease_sum == df_disease_count["Disease_Risk"])

print("Diseases with over 70 samples")
print(df_disease_count[df_disease_count > 70].index)

print("Diseases below 70 samples")
print(df_disease_count[df_disease_count < 70].index)