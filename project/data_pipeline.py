import pandas as pd
import numpy as np

# =============================
# 1️⃣ قراءة البيانات
# =============================
df = pd.read_csv("HDHI Admission data.csv")
print("Initial Data Shape:", df.shape)

# =============================
# 2️⃣ إزالة الأعمدة الفريدة / التعريفية
# =============================
cols_to_drop = ["SNO", "MRD No.", "month year", "D.O.A", "D.O.D"]
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
print("Dropped unique/identifier columns. Current shape:", df.shape)

# =============================
# 3️⃣ التعامل مع الأعمدة الرقمية
# =============================
numeric_cols = ["AGE", "DURATION OF STAY", "duration of intensive unit stay",
                "SMOKING", "ALCOHOL", "DM", "HTN", "CAD", "PRIOR CMP",
                "CKD", "HB", "TLC", "PLATELETS", "GLUCOSE", "UREA",
                "CREATININE", "BNP", "EF"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Warning: Column {col} not found, skipping.")

existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
df[existing_numeric_cols] = df[existing_numeric_cols].fillna(df[existing_numeric_cols].median())
print("Numeric columns cleaned.")

# =============================
# 4️⃣ إزالة outliers باستخدام IQR
# =============================
def remove_outliers(df, columns):
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

df = remove_outliers(df, existing_numeric_cols)
print("Outliers removed. Current shape:", df.shape)

# =============================
# 5️⃣ تحويل الأعمدة الفئوية لـ One-Hot Encoding
# =============================
categorical_cols = ["GENDER", "RURAL", "TYPE OF ADMISSION-EMERGENCY/OPD",
                    "OUTCOME", "CHEST INFECTION"]

existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
df = pd.get_dummies(df, columns=existing_categorical_cols, prefix_sep='_', drop_first=False)
print("Categorical columns encoded.")

# =============================
# 6️⃣ تحويل الأعمدة True/False لـ 0/1
# =============================
for col in df.columns:
    # لو العمود bool
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)
    # لو العمود object ويحتوي على نص "True"/"False"
    elif df[col].dtype == 'object':
        df[col] = df[col].replace({'True': 1, 'False': 0})
        df[col] = df[col].fillna(0)  # لو فيه NaN نخليه 0

print("Boolean columns converted to 0/1.")

# =============================
# 7️⃣ التأكد من أنواع الأعمدة النهائية
# =============================
df = df.apply(pd.to_numeric, errors='ignore')
print("Data preprocessing completed successfully.")
print("Final shape:", df.shape)

# =============================
# 8️⃣ حفظ البيانات بعد التنظيف (اختياري)
# =============================
df.to_csv("HDHI_Admission_Cleaned.csv", index=False)
print("Cleaned dataset saved to HDHI_Admission_Cleaned.csv")
