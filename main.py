import pandas as pd
import numpy as np


def read_data() -> pd.DataFrame:
    df = pd.read_csv("datasets/covid_data.csv")

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

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

    df.loc[:, bool_columns] = df.loc[:, bool_columns].replace(2, 0)
    df.loc[:, bool_columns] = df.loc[:, bool_columns].replace(97, np.nan)
    df.loc[:, bool_columns] = df.loc[:, bool_columns].replace(98, np.nan)
    df.loc[:, bool_columns] = df.loc[:, bool_columns].replace(99, np.nan)
    df.loc[:, "DATE_DIED"] = np.where(df.loc[:, "DATE_DIED"] == "9999-99-99", 0, 1)

    df = df.dropna()

    return df


if __name__ == "__main__":
    df = read_data()
    df = preprocess_data(df)

    print(df.head(10))
