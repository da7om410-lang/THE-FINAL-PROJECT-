import pandas as pd      # استيراد مكتبة pandas لمعالجة البيانات (جداول، CSV، تنظيف)
import numpy as np       # استيراد numpy للعمليات الرياضية (غير مستخدمة كثير هنا لكن مفيدة)
import os                # مكتبة للتعامل مع الملفات والمسارات
from tkinter import Tk   # استيراد Tk لإنشاء نافذة بسيطة
from tkinter.filedialog import askopenfilename  # أداة لاختيار ملف يدويًا

# ============================
# 1) Choose File Manually
# ============================

Tk().withdraw()
# إخفاء نافذة Tkinter الرئيسية (نخلي بس نافذة اختيار الملف)

file_path = askopenfilename(
    title="اختر ملف البيانات (CSV)",
    filetypes=[("CSV files", "*.csv")]
)
# فتح نافذة تسمح لك تختار ملف CSV من جهازك

if not file_path:
    raise ValueError("❌ لم يتم اختيار أي ملف")
# إذا المستخدم ما اختار ملف → يوقف البرنامج ويعطي خطأ

# ============================
# 2) Load Data
# ============================

df = pd.read_csv(file_path)
# قراءة الملف وتحويله إلى DataFrame (جدول بيانات)

print(df.head())
# عرض أول 5 صفوف عشان تفهم شكل البيانات

print("Shape:", df.shape)
# طباعة عدد الصفوف والأعمدة (rows, columns)

print(df.info())
# عرض معلومات عن الأعمدة (نوع البيانات + القيم المفقودة)

# ============================
# 3) Fix Data Types
# ============================

df["MS SubClass"] = df["MS SubClass"].astype(int)
# تحويل العمود إلى أرقام صحيحة (int)

df["Yr Sold"] = df["Yr Sold"].astype(int)
# تحويل سنة البيع إلى عدد صحيح

# ============================
# 4) Missing Values
# ============================

num_cols = df.select_dtypes(include=["number"]).columns
# تحديد الأعمدة الرقمية (أرقام)

cat_cols = df.select_dtypes(exclude=["number"]).columns
# تحديد الأعمدة النصية (كلمات)

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
    # تعويض القيم المفقودة في الأعمدة الرقمية بالوسيط (median)

for col in cat_cols:
    mode_val = df[col].mode(dropna=True)
    # إيجاد أكثر قيمة مكررة (mode)

    df[col] = df[col].fillna(mode_val[0] if len(mode_val) else "Unknown")
    # تعويض القيم المفقودة:
    # إذا فيه قيمة متكررة → نستخدمها
    # إذا لا → نحط "Unknown"

# ============================
# 5) Remove Duplicates
# ============================

df = df.drop_duplicates()
# حذف الصفوف المكررة من البيانات

# ============================
# 6) Outliers (99th percentile)
# ============================

upper = df["SalePrice"].quantile(0.99)
# حساب الحد الأعلى (أعلى 1% من القيم)

df["SalePrice"] = df["SalePrice"].clip(upper=upper)
# قص القيم الكبيرة جدًا (Outliers) عند هذا الحد

# ============================
# 7) Final Checks
# ============================

assert df["SalePrice"].isna().sum() == 0
# التأكد أنه ما فيه قيم مفقودة في السعر

assert (df["SalePrice"] > 0).all()
# التأكد أن كل الأسعار أكبر من صفر

# ============================
# 8) Save Clean File
# ============================

output_dir = "data/cleaned"
# تحديد مجلد الحفظ

os.makedirs(output_dir, exist_ok=True)
# إنشاء المجلد إذا ما كان موجود

output_path = os.path.join(output_dir, "ames_clean.csv")
# إنشاء المسار الكامل للملف

df.to_csv(output_path, index=False)
# حفظ البيانات بعد التنظيف بدون رقم الصف

print("✔ Saved cleaned dataset at:", output_path)
# طباعة رسالة نجاح مع مكان حفظ الملف
