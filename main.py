import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
    df = df.rename(columns={"DATE_DIED": "PATIENT_SURVIVAL"})

    df = df.dropna()

    return df


def train_model(df):
    y = df.pop("PATIENT_SURVIVAL").astype("int")
    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    clf = make_pipeline(
        StandardScaler(), LinearSVC(random_state=0, tol=1e-5, verbose=1)
    )
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"Score of the classifier: {score}")


if __name__ == "__main__":
    df = read_data()
    df = preprocess_data(df)

    train_model(df)
