# ==============================================
# BONUS ALL-IN-ONE SCRIPT
# ==============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# ------------------------------------------------
# 1) Load Ames Dataset (مشروعك الأساسي)
# ------------------------------------------------
df = pd.read_csv("data/cleaned/ames_features_final.csv")

# تحديد اسم عمود السعر (مرن)
price_col = "SalePrice" if "SalePrice" in df.columns else "SalePrice_Log"

# ------------------------------------------------
# 2) Dashboard (4 Charts in One Figure)
# ------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# 1) Histogram
sns.histplot(df[price_col], kde=True, ax=ax[0,0])
ax[0,0].set_title("Price Distribution")

# 2) Scatter
x_col = "Gr Liv Area" if "Gr Liv Area" in df.columns else "GrLivArea_scaled"
sns.scatterplot(x=x_col, y=price_col, data=df, ax=ax[0,1])
ax[0,1].set_title("Area vs Price")

# 3) Boxplot
if "OverallQual_Ord" in df.columns:
    sns.boxplot(x="OverallQual_Ord", y=price_col, data=df, ax=ax[1,0])
    ax[1,0].set_title("Quality vs Price")

# 4) Heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, ax=ax[1,1])
ax[1,1].set_title("Correlation Matrix")

plt.tight_layout()
plt.show()

# ------------------------------------------------
# 3) Deep Analysis (Top Features)
# ------------------------------------------------
print("\n===== Top Features Correlated with Price =====")

corr_target = (
    df.corr(numeric_only=True)[price_col]
    .sort_values(ascending=False)
)

print(corr_target.head(10))

# ------------------------------------------------
# 4) Load Alternative Dataset (California Housing)
# ------------------------------------------------
data = fetch_california_housing(as_frame=True)
df2 = data.frame

print("\n===== California Dataset Preview =====")
print(df2.head())

# ------------------------------------------------
# 5) Analyze Second Dataset
# ------------------------------------------------
print("\n===== Top Features in California Dataset =====")

corr2 = (
    df2.corr(numeric_only=True)["MedHouseVal"]
    .sort_values(ascending=False)
)

print(corr2.head(10))

# ------------------------------------------------
# 6) Simple Comparison Output
# ------------------------------------------------
print("\n===== Comparison =====")

print("Ames Top Feature:", corr_target.index[1])
print("California Top Feature:", corr2.index[1])

print("\nInsight:")
print("- Ames يعتمد أكثر على حجم المنزل والجودة")
print("- California يعتمد أكثر على دخل المنطقة (MedInc)")

# ------------------------------------------------
# DONE
# ------------------------------------------------
print("\n✔ BONUS COMPLETED (5/5)")
