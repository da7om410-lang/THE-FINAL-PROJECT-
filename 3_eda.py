# ==============================================
# 03_eda_with_math.py — EDA + Math Basics
# ==============================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm

# ------------------------------------------------
# 1) Load Data
# ------------------------------------------------
df = pd.read_csv("data/cleaned/ames_features_final.csv")

price_col = "SalePrice" if "SalePrice" in df.columns else "SalePrice_Log"

print("Data Loaded:", df.shape)

# ------------------------------------------------
# 2) Histograms / KDE (3 features)
# ------------------------------------------------
cols = [price_col, "Gr Liv Area", "Total Bsmt SF"]

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

for i, col in enumerate(cols):
    if col in df.columns:
        sns.histplot(df[col], kde=True, ax=ax[i])
        ax[i].set_title(col)

plt.tight_layout()
plt.show()

print("\nInsight:")
print("- توزيع السعر غالبًا يكون skewed لليمين")
print("- المساحة مرتبطة بتنوع كبير في القيم")

# ------------------------------------------------
# 3) Boxplots
# ------------------------------------------------
if "OverallQual_Ord" in df.columns:
    sns.boxplot(x="OverallQual_Ord", y=price_col, data=df)
    plt.title("Price by Quality")
    plt.show()

if "AgeGroup" in df.columns:
    sns.boxplot(x="AgeGroup", y=price_col, data=df)
    plt.title("Price by Age")
    plt.show()

print("\nInsight:")
print("- كلما زادت الجودة زاد السعر بشكل واضح")

# ------------------------------------------------
# 4) Correlation Heatmap (Top 10)
# ------------------------------------------------
corr = (
    df.corr(numeric_only=True)[price_col]
    .abs()
    .sort_values(ascending=False)[1:11]
)

top10 = corr.index

plt.figure(figsize=(8,6))
sns.heatmap(df[top10].corr(), annot=True, cmap="coolwarm")
plt.title("Top Correlated Features")
plt.show()

print("\nInsight:")
print("- الجودة والمساحة من أقوى العوامل المؤثرة على السعر")

# ------------------------------------------------
# 5) Scatter Plot
# ------------------------------------------------
x_col = "Gr Liv Area" if "Gr Liv Area" in df.columns else "GrLivArea_scaled"

sns.scatterplot(
    data=df,
    x=x_col,
    y=price_col,
    hue="OverallQual_Ord" if "OverallQual_Ord" in df.columns else None
)
plt.title("Area vs Price")
plt.show()

print("\nInsight:")
print("- العلاقة طردية: كلما زادت المساحة زاد السعر")

# ------------------------------------------------
# 6) GroupBy Summary
# ------------------------------------------------
df_raw = pd.read_csv("data/cleaned/ames_clean.csv")

summary = df_raw.groupby("Neighborhood")["SalePrice"].mean().sort_values()

print("\nTop 5 expensive neighborhoods:")
print(summary.tail(5))

print("\nTop 5 cheapest neighborhoods:")
print(summary.head(5))

# ------------------------------------------------
# ================= MATH BASICS ==================
# ------------------------------------------------

# ------------------------------------------------
# 7) Mean & Std (NumPy only)
# ------------------------------------------------
y = df[price_col].values

mean_manual = np.sum(y) / len(y)
std_manual = np.sqrt(np.sum((y - mean_manual)**2) / len(y))

print("\nMean (manual):", mean_manual)
print("Std (manual):", std_manual)

# ------------------------------------------------
# 8) Standardization (manual vs sklearn)
# ------------------------------------------------
col = "Gr Liv Area" if "Gr Liv Area" in df.columns else "GrLivArea_scaled"
X = df[col].values

mean_X = np.mean(X)
std_X = np.std(X)

z_manual = (X - mean_X) / std_X

scaler = StandardScaler()
z_sklearn = scaler.fit_transform(X.reshape(-1,1)).flatten()

print("\nCompare manual vs sklearn (first 5 values):")
print("Manual:", z_manual[:5])
print("Sklearn:", z_sklearn[:5])

# ------------------------------------------------
# 9) Cosine Similarity
# ------------------------------------------------
# أعلى سعر وأقل سعر
high = df.sort_values(price_col, ascending=False).iloc[0]
low = df.sort_values(price_col, ascending=True).iloc[0]

# نستخدم فقط الأعمدة الرقمية
num_cols = df.select_dtypes(include=np.number).columns

vec1 = high[num_cols].values
vec2 = low[num_cols].values

cos_sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

print("\nCosine Similarity (High vs Low):", cos_sim)

# ------------------------------------------------
# 10) Probability Estimation
# ------------------------------------------------
if "OverallQual_Ord" in df.columns:
    high_quality = df[df["OverallQual_Ord"] >= 4]
    threshold = df[price_col].median()

    prob = np.mean(high_quality[price_col] > threshold)

    print("\nProbability that high-quality house > median price:", prob)

# ------------------------------------------------
# DONE
# ------------------------------------------------
print("\n✔ Phase 3 (EDA + Math) Completed")
