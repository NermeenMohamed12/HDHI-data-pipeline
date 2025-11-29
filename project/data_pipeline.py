

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("HDHI Admission data.csv")

print("Initial Data Shape:", df.shape)
print(df.head())


# =======================================================
# 2) Handle Missing Values
# =======================================================

# Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled successfully.")


# =======================================================
# 3) Remove Outliers for AGE + DURATION OF STAY
# =======================================================

def clean_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

df = clean_outliers_iqr(df, "AGE")
df = clean_outliers_iqr(df, "DURATION OF STAY")

print("Outliers removed. Current shape:", df.shape)


# =======================================================
# 4) ENCODING CATEGORICAL COLUMNS
# =======================================================

# Columns in your dataset:
#   - GENDER (M/F)
#   - RURAL (R/U)
#   - TYPE OF ADMISSION-EMERGENCY/OPD (E/OPD)

df = pd.get_dummies(
    df,
    columns=["GENDER", "RURAL", "TYPE OF ADMISSION-EMERGENCY/OPD"],
    drop_first=True
)

print("Encoding completed. Current shape:", df.shape)


# =======================================================
# 5) SPLITTING: Train (70%) - Validation (15%) - Test (15%)
# =======================================================

train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print("Train shape:", train.shape)
print("Validation shape:", val.shape)
print("Test shape:", test.shape)


# =======================================================
# 6) SAVE CLEANED DATA
# =======================================================

train.to_csv("train_data.csv", index=False)
val.to_csv("val_data.csv", index=False)
test.to_csv("test_data.csv", index=False)

print("-----------------------------------------------------")
print("DATA PIPELINE COMPLETED SUCCESSFULLY")
print("Files saved: train_data.csv, val_data.csv, test_data.csv")
print("-----------------------------------------------------")
