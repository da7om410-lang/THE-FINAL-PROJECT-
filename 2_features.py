# ==============================================
# 02_features.py — Phase 2: Feature Engineering (FIXED)
# ==============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# 1) Load cleaned dataset from Phase 1
# ------------------------------------------------
df = pd.read_csv("data/cleaned/ames_clean.csv")
print("Initial shape:", df.shape)

# ------------------------------------------------
# 2) One-hot Encoding (2 categorical columns)
# ------------------------------------------------
df = pd.get_dummies(
    df,
    columns=["Neighborhood", "House Style"],
    drop_first=True
)
print("After One-Hot Encoding:", df.shape)

# ------------------------------------------------
# 3) Ordinal Encoding (1 ordered column)
# ------------------------------------------------
quality_map = {"Poor":1, "Fair":2, "Average":3, "Good":4, "Excellent":5}
df["OverallQual_Ord"] = df["Overall Qual"].replace(quality_map)
print("Added Ordinal Feature")

# ------------------------------------------------
# 4) Scaling (2 numerical columns)
# ------------------------------------------------
scaler = StandardScaler()
df[["GrLivArea_scaled", "TotalBsmtSF_scaled"]] = scaler.fit_transform(
    df[["Gr Liv Area", "Total Bsmt SF"]]
)
print("Added Scaled Features")

# ------------------------------------------------
# 5) Domain Features (safe division)
# ------------------------------------------------
def safe_div(a, b):
    return a / b if b != 0 else 0

df["price_per_sqft"] = [
    safe_div(s, a) for s, a in zip(df["SalePrice"], df["Gr Liv Area"])
]

df["house_age_ratio"] = [
    safe_div(h, r) for h, r in zip(
        df["Yr Sold"] - df["Year Built"],
        df["Yr Sold"] - df["Year Remod/Add"]
    )
]
print("Added Domain Features")

# ------------------------------------------------
# 6) Interaction Feature
# ------------------------------------------------
df["Qual_Area_Interaction"] = df["OverallQual_Ord"] * df["Gr Liv Area"]
print("Added Interaction Feature")

# ------------------------------------------------
# 7) Log-transform SalePrice (بدون حذف الأصل)
# ------------------------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["SalePrice"], kde=True)
plt.title("SalePrice Before Log")
plt.show()

df["SalePrice_Log"] = np.log1p(df["SalePrice"])

plt.figure(figsize=(6,4))
sns.histplot(df["SalePrice_Log"], kde=True)
plt.title("SalePrice After Log")
plt.show()

print("Added SalePrice_Log")

# ------------------------------------------------
# 8) Binning Age into Groups
# ------------------------------------------------
age = df["Yr Sold"] - df["Year Built"]

df["AgeGroup"] = pd.cut(
    age,
    bins=[0, 10, 30, 200],
    labels=["New", "Recent", "Old"],
    include_lowest=True
)
print("Added AgeGroup Feature")

# ------------------------------------------------
# 9) Remove Highly Correlated Features (SAFE)
# ------------------------------------------------
corr_matrix = df.corr(numeric_only=True).abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

cols_to_drop = [
    col for col in upper.columns
    if any(upper[col] > 0.95)
]

# 🔥 نحمي الأعمدة المهمة
protected_cols = ["SalePrice", "SalePrice_Log"]

cols_to_drop = [
    col for col in cols_to_drop
    if col not in protected_cols
]

df = df.drop(columns=cols_to_drop, errors="ignore")

print("Dropped columns:", cols_to_drop)

# ------------------------------------------------
# 10) Final Safety Check
# ------------------------------------------------
if "SalePrice" not in df.columns:
    raise ValueError("❌ SalePrice was removed by mistake!")

# ------------------------------------------------
# 11) Save Final Engineered File
# ------------------------------------------------
df.to_csv("data/cleaned/ames_features_final.csv", index=False)
print("\n✔ Final Engineered File Saved:", df.shape)
