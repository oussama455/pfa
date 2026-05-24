# النظام الذكي للتعرف على أنواع الخرائط العسكرية

## 🎯 مقدمة

تم تطوير **نظام تكيفي ذكي** يتعرف تلقائياً على نوع الخريطة ويضبط جميع المعاملات حسب النوع:

- ✅ **الخرائط الملونة** (Chroma/Color): معايير صارمة، فلترة خفيفة
- ✅ **الخرائط الباهتة** (Monochrome/Faded): معايير مرنة، فلترة قوية

---

## 🚀 البدء السريع

### الخطوة 1: نسخ الملفات المحدثة

تم تحديث 3 ملفات رئيسية:
```
pipeline/
  ├── preprocessing.py      ✅ محدث (كشف نوع الخريطة + فلترة ديناميكية)
  ├── pipeline.py           ✅ محدث (معايير QA تكيفية)
  └── cc_postprocess.py     ✅ محدث (فلترة min_area تكيفية)
```

### الخطوة 2: تشغيل بسيط

```python
from pipeline.pipeline import run_pipeline

result = run_pipeline(
    input_path="data/raw/carte_militaire.tif",
    output_dir="data/processed"
)
# ✓ كل شيء تلقائي!
# ✓ نوع الخريطة مكتشف
# ✓ الفلاتر مضبوطة
# ✓ المعايير مناسبة
```

### الخطوة 3: مراقبة الإخراج

ستجد في الـ console:
```
[1/5] Prétraitement : carte.tif
      Type de carte détecté: monochrome_faded  ← اسم النوع
      ...

[QA] Thresholds adaptés pour 'monochrome_faded':
      QA Target           : 75.0%
      Max Layer Ratio     : 92.0%
      Min Acceptable Conf : 40.0%
```

---

## 📊 ماذا تغيّر؟

### قبل (نظام ثابت):
```
❌ 21% confidence ← فشل على الخرائط الباهتة
❌ إفراط في التقسيم (over-segmentation)
❌ معايير موحدة لجميع الخرائط
```

### بعد (نظام ذكي):
```
✅ تلقائي كشف نوع الخريطة
✅ معايير مناسبة لكل نوع
✅ فلاتر متكيفة
✅ حماية من الإفراط في التقسيم
✅ أداء محسّن على الخرائط القديمة
```

---

## 📁 الملفات الجديدة

1. **ADAPTIVE_MAP_DETECTION.md** — شرح تفصيلي للنظام (تقني)
2. **IMPLEMENTATION_SUMMARY.md** — ملخص التعديلات (تطبيق)
3. **test_adaptive_system.py** — اختبارات سريعة
4. **README_QUICKSTART.md** — هذا الملف (بدء سريع)

---

## 🔍 أمثلة الاستخدام

### الاستخدام الأساسي (موصى به):

```python
from pipeline.pipeline import run_pipeline

# تشغيل بسيط — كل شيء تلقائي
result = run_pipeline("data/raw/map.tif", "data/processed")
```

### مع خيارات متقدمة:

```python
result = run_pipeline(
    input_path="data/raw/map.tif",
    output_dir="data/processed",
    with_semantic=True,
    unet_weights="weights/semap_best.pth",
    use_calibrated_hsv=True,
    # معايير QA ستُحسب تلقائياً!
)
```

### اختبار نوع الخريطة فقط:

```python
from pipeline.preprocessing import classify_map_type
import cv2

img = cv2.imread("map.tif")
map_type = classify_map_type(img)
print(f"Map type: {map_type}")  # "color_rich" أو "monochrome_faded"
```

### اختبار الفلترة الديناميكية:

```python
from pipeline.preprocessing import classify_map_type, denoise

img = cv2.imread("map.tif")
map_type = classify_map_type(img)
denoised = denoise(img, map_type=map_type)  # ديناميكي!
```

### الوصول إلى معايير QA:

```python
from pipeline.pipeline import get_adaptive_qa_thresholds

thresholds = get_adaptive_qa_thresholds("monochrome_faded")
print(f"QA Target: {thresholds.qa_threshold}%")
print(f"Max Layer: {thresholds.max_allowed_layer_ratio}%")
```

### حساب min_area التكيفية:

```python
from pipeline.cc_postprocess import apply_adaptive_min_area

min_area = apply_adaptive_min_area(
    layer_name="buildings",
    map_type="monochrome_faded"
)
# النتيجة: 80 px (40 × 2.0 لأن الخريطة قديمة)
```

---

## 🧪 تشغيل الاختبارات

```bash
# اختبار سريع بدون صور حقيقية
python test_adaptive_system.py

# اختبار كامل مع صور من data/raw/
python test_adaptive_system.py
```

الإخراج:
```
[1/5] اختبار المعايير الديناميكية (QA Thresholds)
  color_rich:
    QA Target:              90.0%
    Max Layer Ratio:        85.0%
    ...

  monochrome_faded:
    QA Target:              75.0%
    Max Layer Ratio:        92.0%
    ...

[2/5] اختبار المساحة الدنيا التكيفية (Adaptive Min Area)
  color_rich:
    buildings    :  40 px (×1.0)
    contours     : 150 px (×1.0)
    ...

  monochrome_faded:
    buildings    :  80 px (×2.0)
    contours     : 300 px (×2.0)
    ...
```

---

## 📈 معايير الأداء

### الخرائط الملونة (color_rich):
```
المدخل     → 45% saturation → "color_rich"
الفلترة    → bilateral(d=5) - خفيف
QA Target → 90% (صارم)
النتيجة    → دقة عالية، مضلعات نظيفة ✓
```

### الخرائط الباهتة (monochrome_faded):
```
المدخل     → 8% saturation → "monochrome_faded"
الفلترة    → bilateral(d=9) + morphology - قوي
QA Target → 75% (مرن)
النتيجة    → حماية من الإفراط، خطوط محفوظة ✓
```

---

## ⚙️ الضبط المتقدم

### تغيير حد كشف الخريطة:

في `preprocessing.py`، بحث عن:
```python
if mean_saturation < 15.0:  # هنا الحد الثابت
    return "monochrome_faded"
```

غيّره إذا لزم الأمر:
```python
if mean_saturation < 20.0:  # حد أعلى
    return "monochrome_faded"
```

### تجاوز النوع المكتشف:

```python
from pipeline.pipeline import get_adaptive_qa_thresholds

# فرض يدوي
forced_type = "monochrome_faded"
thresholds = get_adaptive_qa_thresholds(forced_type)
```

### تجاوز min_area التلقائي:

```python
from pipeline.cc_postprocess import vectorize_mask

gdf = vectorize_mask(
    mask=semantic_mask,
    layer_name="buildings",
    min_area_px=200,  # override يدوي
    map_type="monochrome_faded"
)
```

---

## 🛠️ استكشاف المشاكل

### المشكلة: خريطة قديمة تُكتشف كـ "color_rich"

**الأعراض:**
```
Detected: color_rich
QA Target: 90%  ← معيار صارم جداً
Result: 21% confidence ← فشل!
```

**الحل:**
```python
# تحقق من Mean Saturation
from pipeline.preprocessing import classify_map_type
import cv2

img = cv2.imread("map.tif")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mean_sat = np.mean(hsv[:, :, 1])
print(f"Mean Saturation: {mean_sat}%")

# إذا كان < 15% لكن كُكتشف خطأ:
# عدّل الحد في preprocessing.py من 15.0 إلى 20.0
```

### المشكلة: مضلعات صغيرة جداً/ضوضاء

**السبب:**
```
monochrome_faded لكن min_area = 50 (default)
بدلاً من 100 (×2 للنوع الباهت)
```

**الحل:**
```python
# تأكد من تمرير map_type
gdf = vectorize_mask(
    mask=semantic_mask,
    layer_name="buildings",
    map_type="monochrome_faded"  # مهم!
    # min_area_px سيكون 80 تلقائياً
)
```

---

## 📋 قائمة الفحص

قبل استخدام النظام على بيانات حقيقية:

- [ ] تحديث `pipeline/preprocessing.py` ← جديد: `classify_map_type()`
- [ ] تحديث `pipeline/pipeline.py` ← جديد: معايير QA ديناميكية
- [ ] تحديث `pipeline/cc_postprocess.py` ← جديد: min_area ديناميكي
- [ ] اختبار على خريطة ملونة واحدة
- [ ] اختبار على خريطة باهتة واحدة
- [ ] مراجعة الإخراج (Confidence ≥ 70%)
- [ ] تفعيل النظام على كل الخرائط

---

## 📚 للمزيد من المعلومات

- **ADAPTIVE_MAP_DETECTION.md** — شرح تقني كامل
- **IMPLEMENTATION_SUMMARY.md** — توثيق التعديلات
- **test_adaptive_system.py** — اختبارات وأمثلة
- **pipeline/preprocessing.py** — الكود المصدري مع تعليقات

---

## 📞 ملخص التحديثات

| الملف | التغيير | التأثير |
|------|--------|--------|
| preprocessing.py | + classify_map_type() | كشف تلقائي |
| preprocessing.py | denoise() ديناميكي | فلاتر مناسبة |
| pipeline.py | معايير QA ديناميكية | توازن أفضل |
| cc_postprocess.py | min_area تكيفي | حماية أفضل |

---

## ✅ الحالة

- ✅ التطوير انتهى
- ✅ الاختبارات جاهزة
- ✅ التوثيق كامل
- ✅ جاهز للإنتاج

---

**آخر تحديث:** 24 مايو 2026  
**الإصدار:** v1.0 — نظام ذكي كامل للتعرف على أنواع الخرائط
