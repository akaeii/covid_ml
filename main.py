import pandas as pd
import numpy as np

df = pd.read_csv("datasets/covid_data.csv")

df = df[
    [
        "USMER",
        "SEX",
        "PATIENT_TYPE",
        "DATE_DIED",
        "INTUBED",
        "PNEUMONIA",
        "AGE",
        "PREGNANT",
        "DIABETES",
        "COPD",
        "ASTHMA",
        "INMSUPR",
        "HIPERTENSION",
        "OTHER_DISEASE",
        "CARDIOVASCULAR",
        "OBESITY",
        "RENAL_CHRONIC",
        "TOBACCO",
        "CLASIFFICATION_FINAL",
        "ICU",
    ]
]

print(df.head())

bool_columns = [
    "SEX",
    "PATIENT_TYPE",
    "INTUBED",
    "PNEUMONIA",
    "PREGNANT",
    "DIABETES",
    "COPD",
    "ASTHMA",
    "INMSUPR",
    "HIPERTENSION",
    "OTHER_DISEASE",
    "CARDIOVASCULAR",
    "OBESITY",
    "RENAL_CHRONIC",
    "TOBACCO",
    "ICU",
]

df[bool_columns] = df[bool_columns].replace(2, 0)
df[bool_columns] = df[bool_columns].replace(97, np.nan)
df[bool_columns] = df[bool_columns].replace(99, np.nan)

df["DATE_DIED"] = np.where(df["DATE_DIED"] == "9999-99-99", 0, 1)

print(df.head(10))
