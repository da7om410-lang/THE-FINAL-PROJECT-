# ==============================================
# 03_eda.py — Phase 3: Exploratory Data Analysis (FIXED)
# ==============================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1) Load engineered dataset
# ------------------------------------------------
df = pd.read_csv("data/cleaned/ames_features_final.csv")
print("Columns loaded:\n", df.columns.tolist())

# ------------------------------------------------
# 2) Handle column name differences (SAFE)
# ------------------------------------------------
# بعض الأعمدة ممكن تكون اتغيرت
col_map = {
    "Gr Liv Area": "GrLivArea_scaled" if "GrLivArea_scaled" in df.columns else "Gr Liv Area",
    "Total Bsmt SF": "TotalBsmtSF_scaled" if "TotalBsmtSF_scaled" in df.columns else "Total Bsmt SF",
    "SalePrice": "SalePrice" if "SalePrice" in df.columns else "SalePrice_Log"
}

# تأكد من الأعمدة
num_cols = [
    col_map["SalePrice"],
    col_map["Gr Liv Area"],
    col_map["Total Bsmt SF"]
]

# ------------------------------------------------
# 3) Histograms / KDE
# ------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

for i, col in enumerate(num_cols):
    if col in df.columns:
        sns.histplot(df[col], kde=True, ax=ax[i])
        ax[i].set_title(f"{col} Distribution")
    else:
        ax[i].set_title(f"{col} NOT FOUND")

plt.tight_layout()
plt.show()

# ------------------------------------------------
# 4) Boxplots — Categories vs Price
# ------------------------------------------------
price_col = col_map["SalePrice"]

if "OverallQual_Ord" in df.columns:
    sns.boxplot(x="OverallQual_Ord", y=price_col, data=df)
    plt.title("Price by Overall Quality")
    plt.show()

if "AgeGroup" in df.columns:
    sns.boxplot(x="AgeGroup", y=price_col, data=df)
    plt.title("Price by Age Group")
    plt.show()

# ------------------------------------------------
# 5) Correlation Heatmap — Top 10
# ------------------------------------------------
if price_col in df.columns:
    corr = (
        df.corr(numeric_only=True)[price_col]
        .abs()
        .sort_values(ascending=False)[1:11]
    )

    top10 = corr.index

    plt.figure(figsize=(8,6))
    sns.heatmap(df[top10].corr(), annot=True, cmap="coolwarm")
    plt.title("Top 10 Features Correlated with Price")
    plt.show()
else:
    print("⚠️ Price column not found for correlation")

# ------------------------------------------------
# 6) Scatterplot
# ------------------------------------------------
x_col = col_map["Gr Liv Area"]

if all(c in df.columns for c in [x_col, price_col]):
    sns.scatterplot(
        data=df,
        x=x_col,
        y=price_col,
        hue="OverallQual_Ord" if "OverallQual_Ord" in df.columns else None
    )
    plt.title("Price vs Living Area")
    plt.show()

# ------------------------------------------------
# 7) GroupBy (Neighborhood analysis)
# ------------------------------------------------
df_raw = pd.read_csv("data/cleaned/ames_clean.csv")

if "Neighborhood" in df_raw.columns:
    neigh_summary = (
        df_raw.groupby("Neighborhood")["SalePrice"]
        .mean()
        .sort_values()
    )

    print("\nTop 10 expensive neighborhoods:")
    print(neigh_summary.tail(10))

    print("\nBottom 10 cheapest neighborhoods:")
    print(neigh_summary.head(10))
else:
    print("⚠️ Neighborhood column not found")

# ------------------------------------------------
# 8) Done
# ------------------------------------------------
print("\n✔ EDA Completed Successfully")
