import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =============================
# 1️⃣ قراءة البيانات
# =============================
df = pd.read_csv("HDHI Admission data.csv")
print("Initial Data Shape:", df.shape)

# =============================
# 2️⃣ إزالة الأعمدة الفريدة / التعريفية
# =============================
cols_to_drop = ["SNO", "MRD No.", "month year"]
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
print("Dropped unique/identifier columns. Current shape:", df.shape)

# =============================
# 3️⃣ التعامل مع الأعمدة الرقمية
# =============================
numeric_cols = ["AGE", "DURATION OF STAY", "duration of intensive unit stay",
                "SMOKING", "ALCOHOL", "HB", "TLC", "PLATELETS",
                "GLUCOSE", "UREA", "CREATININE", "BNP", "EF"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

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
# 5️⃣ تحويل الأعمدة الفئوية لـ One-Hot Encoding (باستثناء OUTCOME)
# =============================
categorical_cols = ["GENDER", "RURAL", "TYPE OF ADMISSION-EMERGENCY/OPD", "CHEST INFECTION"]
existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
df = pd.get_dummies(df, columns=existing_categorical_cols, prefix_sep='_', drop_first=False)
print("Categorical columns encoded.")

# =============================
# 6️⃣ تحويل True/False لـ 0/1
# =============================
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)
    elif df[col].dtype == 'object':
        df[col] = df[col].replace({'True': 1, 'False': 0})
        df[col] = df[col].fillna(0)

print("Boolean columns converted to 0/1.")

# =============================
# 7️⃣ تحويل OUTCOME من string لأرقام
# =============================
if 'OUTCOME' in df.columns:
    le = LabelEncoder()
    df['OUTCOME'] = le.fit_transform(df['OUTCOME'])
    print("OUTCOME converted to numeric labels:", dict(zip(le.classes_, le.transform(le.classes_))))

# =============================
# 8️⃣ تحديد الأعمدة المرضية كـ target
# =============================
disease_columns = ["DM", "HTN", "CAD", "PRIOR CMP", "CKD", "SEVERE ANAEMIA",
                   "ANAEMIA", "STABLE ANGINA", "ACS", "STEMI", "ATYPICAL CHEST PAIN",
                   "HEART FAILURE", "HFREF", "HFNEF", "VALVULAR", "CHB", "SSS",
                   "AKI", "CVA INFRACT", "CVA BLEED", "AF", "VT", "PSVT",
                   "CONGENITAL", "UTI", "NEURO CARDIOGENIC SYNCOPE", "ORTHOSTATIC",
                   "INFECTIVE ENDOCARDITIS", "DVT", "CARDIOGENIC SHOCK", "SHOCK",
                   "PULMONARY EMBOLISM"]

# تأكد أن الأعمدة موجودة
disease_columns = [col for col in disease_columns if col in df.columns]

X = df.drop(columns=disease_columns)
y = df[disease_columns]

# =============================
# 9️⃣ تقسيم البيانات إلى ثلاث مجموعات (Train 70%, Validation 15%, Test 15%)
# =============================
# تقسيم Test 15%
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=df['OUTCOME'] if 'OUTCOME' in df.columns else None
)

# تقسيم الباقي لـ Train 70% و Validation 15%
val_size = 0.176  # 0.176 من 85% ≈ 15% من الداتا الأصلية
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_size, random_state=42,
    stratify=y_temp[disease_columns[0]] if disease_columns else None
)

# إنشاء DataFrames منفصلة لكل مجموعة
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

print("Training set shape:", train_df.shape)
print("Validation set shape:", val_df.shape)
print("Test set shape:", test_df.shape)

# =============================
# 🔟 حفظ الملفات النهائية
# =============================
train_df.to_csv("HDHI_Train.csv", index=False)
val_df.to_csv("HDHI_Validation.csv", index=False)
test_df.to_csv("HDHI_Test.csv", index=False)
print("DataFrames saved: HDHI_Train.csv, HDHI_Validation.csv, HDHI_Test.csv")
