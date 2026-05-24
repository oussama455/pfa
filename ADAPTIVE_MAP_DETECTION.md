# النظام الذكي للتعرف على أنواع الخرائط العسكرية التاريخية

## نظرة عامة

تم تحديث النظام الكامل ليصبح **ذكياً وتكيفياً** — يكتشف تلقائياً نوع الخريطة ويضبط كل العمليات بناءً عليه.

الآن النظام يتعامل بكفاءة مع:
- **الخرائط الملونة الحديثة** (Chroma/Rich Color): تباين لوني واضح، ألوان زاهية
- **الخرائط الباهتة القديمة** (Monochrome/Faded): ألوان متقاربة، رمادي/بني، ضوضاء من المسح الضوئي

---

## 1. كيفية عمل النظام الجديد

### 1.1 مرحلة التصنيف الآلي (Auto-Classification)

```python
def classify_map_type(img_bgr: np.ndarray) -> str:
```

عند تحميل الخريطة، النظام يقيس **متوسط التشبع اللوني (Mean Saturation)** في فضاء HSV:

- **متوسط التشبع < 15%** → الخريطة **باهتة/أحادية** (`monochrome_faded`)
- **متوسط التشبع ≥ 15%** → الخريطة **ملونة غنية** (`color_rich`)

**مثال:**
```
Bizerte (خريطة ملونة):   Mean Saturation = 45% ✓ color_rich
Tunis (خريطة باهتة):      Mean Saturation = 8%  ✓ monochrome_faded
```

---

## 2. التعديلات على كل مرحلة

### 2.1 مرحلة المعالجة المسبقة (`preprocessing.py`)

#### دالة `denoise()` - فلترة ديناميكية

```python
def denoise(img: np.ndarray, map_type: str = "color_rich") -> np.ndarray:
```

**للخرائط الملونة** (`color_rich`):
```
bilateral_filter(d=5, sigmaColor=50, sigmaSpace=50)
→ تنعيم خفيف يحافظ على تفاصيل المباني والطرق الدقيقة
```

**للخرائط الباهتة** (`monochrome_faded`):
```
bilateral_filter(d=9, sigmaColor=90, sigmaSpace=90)  # قوي جداً
+ MORPH_CLOSE(kernel=2×2)                          # لربط الخطوط المتقطعة
→ إزالة ضوضاء المسح الضوئي مع حماية الخطوط الكنتورية
```

#### تحديث دوال `preprocess()` و `preprocess_with_crop()`

```python
# الآن تعيد 3 قيم (color_rich في السابق):
image_bgr, image_hsv, map_type = preprocess(path)

# الآن تعيد 4 قيم (3 في السابق):
image_bgr, image_hsv, crop_bbox, map_type = preprocess_with_crop(path)
```

---

### 2.2 مرحلة جودة التحكم (QA) - `pipeline.py`

#### الهياكل الجديدة

```python
@dataclass
class AdaptiveQAThresholds:
    qa_threshold: float                 # الهدف (%)
    max_allowed_layer_ratio: float      # أقصى نسبة سيطرة لطبقة واحدة
    min_acceptable_confidence: float    # الحد الأدنى للثقة المقبولة
    max_iterations: int                 # عدد التكرارات المسموح
```

#### المعايير الديناميكية

```python
def get_adaptive_qa_thresholds(map_type: str) -> AdaptiveQAThresholds:
```

**للخرائط الملونة** (`color_rich`):
```
qa_threshold              = 90.0%   (معيار صارم)
max_allowed_layer_ratio   = 85.0%   (عدم تسامح مع عدم التوازن)
min_acceptable_confidence = 55.0%   (معايير عالية)
max_iterations            = 5       (تكرارات أكثر للجودة العالية)
```

**للخرائط الباهتة** (`monochrome_faded`):
```
qa_threshold              = 75.0%   (معيار مرن)
max_allowed_layer_ratio   = 92.0%   (تسامح مع عدم التوازن)
min_acceptable_confidence = 40.0%   (قبول ثقة منخفضة)
max_iterations            = 3       (تكرارات قليلة = تجنب الإفراط)
```

#### حساب الثقة والتوازن

```python
def calculate_layer_coverage_stats(masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    # احسب نسبة تغطية كل طبقة (%)
    
def calculate_confidence_score(coverage_stats, qa_threshold) -> float:
    # درجة ثقة بناءً على توازن الطبقات
    # كلما كانت الطبقات متوازنة → ثقة أعلى
    # إذا هيمنت طبقة واحدة > 90% → ثقة منخفضة (إفراط تقسيم)
```

#### الإخراج أثناء التشغيل

```
[QA] Thresholds adaptés pour 'monochrome_faded':
      QA Target           : 75.0%
      Max Layer Ratio     : 92.0%
      Min Acceptable Conf : 40.0%
      Current Confidence  : 68.5%
      Max Layer Dominance : 78.3%
```

---

### 2.3 مرحلة المكونات المتصلة (`cc_postprocess.py`)

#### دالة الفلترة التكيفية

```python
def apply_adaptive_min_area(layer_name: str, map_type: str = "color_rich") -> int:
```

**للخرائط الملونة** (`color_rich`):
```
buildings  → 40 px    (الحد الأدنى الطبيعي)
contours   → 150 px
water      → 80 px
vegetation → 100 px
roads      → 30 px
```

**للخرائط الباهتة** (`monochrome_faded`) - معامل ×2.0:
```
buildings  → 80 px    (×2.0 = تجاهل الأوساخ الدقيقة)
contours   → 300 px
water      → 160 px
vegetation → 200 px
roads      → 60 px
```

#### استخدام الفلترة في التوجيه

```python
# الاستدعاء التلقائي:
gdf = vectorize_mask(
    mask=semantic_mask,
    layer_name="buildings",
    map_type="monochrome_faded",  # يتم حسابه من preprocessing
    # min_area_px محذوف = حساب تلقائي
)

# أو التحديد اليدوي:
gdf = vectorize_mask(
    mask=semantic_mask,
    layer_name="buildings",
    min_area_px=100,  # override تلقائي
)
```

---

## 3. سير العمل المتكامل

```
[1] صورة المدخل (JPG/TIFF)
     ↓
[2] تحميل + Downscale
     ↓
[3] ⚡ CLASSIFY: تحديد نوع الخريطة تلقائياً
     │    ↓ monochrome_faded?      ↓ color_rich?
     │    strong_denoise()           light_denoise()
     ↓
[4] كشف الإطار (neatline) + حذف الأسطورة (legend)
     ↓
[5] تحويل HSV + تقسيم الألوان
     ↓
[6] ⚡ QA CHECK: معايير ديناميكية
     │    Threshold: 75% أم 90%?
     │    MaxLayerRatio: 92% أم 85%?
     │    MinConfidence: 40% أم 55%?
     ↓
[7] المكونات المتصلة (CC) مع فلترة تكيفية
     │    min_area: تحديد تلقائي بناءً على:
     │    - نوع الطبقة (buildings, contours, water, ...)
     │    - نوع الخريطة (monochrome_faded = ×2.0)
     ↓
[8] تحويل إلى مضلعات + تبسيط
     ↓
[9] GeoJSON/Shapefile
```

---

## 4. أمثلة استخدام

### 4.1 الاستخدام البسيط (المموصى به)

```python
from pipeline.pipeline import run_pipeline

result = run_pipeline(
    input_path="data/raw/biserta_1940.tif",
    output_dir="data/processed",
    # لا تحتاج لتحديد شيء — كل شيء تلقائي!
)

print(f"Detected map type: {detected_map_type}")
print(result.to_json())
```

### 4.2 مع خيارات متقدمة

```python
result = run_pipeline(
    input_path="data/raw/old_faded_map.tif",
    output_dir="data/processed",
    with_semantic=True,
    unet_weights="external/weights/semap_unet_best.pth",
    use_calibrated_hsv=True,
    # معايير QA و min_area تُحسب تلقائياً من نوع الخريطة
)
```

### 4.3 على مرحلة المعالجة المسبقة فقط

```python
from pipeline import preprocessing as prep

image_bgr, image_hsv, map_type = prep.preprocess(
    path="data/raw/map.tif",
    denoise_on=True,  # ديناميكي حسب map_type
)

print(f"Map type detected: {map_type}")
if map_type == "monochrome_faded":
    print("→ Applying STRONG denoising for faded historical map")
```

### 4.4 على مرحلة التوجيه مع خريطة محكمة

```python
from pipeline.cc_postprocess import vectorize_mask

gdf = vectorize_mask(
    mask=unet_predictions,
    layer_name="buildings",
    map_type="monochrome_faded",  # دليل من preprocessing
    # min_area_px سيكون 40 × 2.0 = 80 px
)
```

---

## 5. الحالات المعالجة

### ✅ الخرائط الملونة الحديثة

- **الكشف:** Mean Saturation ≥ 15%
- **المعالجة المسبقة:** فلتر bilateral خفيف (d=5)
- **QA:** معايير صارمة (90% QA)
- **الفلترة:** min_area عادي (مثل 40 px للمباني)
- **النتيجة:** دقة عالية، مضلعات نظيفة

### ✅ الخرائط الباهتة القديمة

- **الكشف:** Mean Saturation < 15%
- **المعالجة المسبقة:** فلتر bilateral قوي (d=9) + مورفولوجي
- **QA:** معايير مرنة (75% QA)
- **الفلترة:** min_area مرتفع (مثل 80 px للمباني)
- **النتيجة:** حماية من الإفراط في التقسيم

### ✅ الخرائط النادرة / المختلطة

إذا كانت الخريطة تجمع بين عناصر ملونة وباهتة:
- النظام يحسب متوسط التشبع الكلي
- يختار النوع الأقرب
- يطبق المعايير المناسبة

---

## 6. معايير التخطيط والضبط

### إذا رأيت "Confidence = 21%" وإفراط تقسيم:

```
السبب المحتمل:
  - خريطة باهتة لم تُكتشف صحيح (monochrome_faded)
  - دخلت معايير color_rich الصارمة (90% QA)
  - نتيجة: محاولة 5 تصحيحات → إفراط تقسيم

الحل:
  - تحقق من Mean Saturation (يجب < 15%)
  - اختبر يدويً: prep.classify_map_type(img)
  - إذا خاطئ: عدّل الحد (15.0 → 20.0)
```

### إذا أردت تجاوز الكشف الآلي:

```python
# لا توجد طريقة حالياً لتمرير map_type يدويًا في run_pipeline
# لكن يمكنك استدعاء المراحل يدويًا:

from pipeline import preprocessing as prep
from pipeline import pipeline as pipe

img, hsv, bbox, detected = prep.preprocess_with_crop(path)

# تجاوز الكشف:
forced_map_type = "monochrome_faded"  # فرض يدوي
thresholds = pipe.get_adaptive_qa_thresholds(forced_map_type)
print(thresholds)
```

---

## 7. الأداء

### الخرائط الملونة (color_rich):
- ✓ معالجة أسرع (فلتر أخف)
- ✓ دقة عالية (90% QA target)
- ✓ مضلعات أنظف

### الخرائط الباهتة (monochrome_faded):
- ⚠ معالجة أبطأ قليلاً (فلتر قوي + مورفولوجي)
- ✓ حماية من الإفراط في التقسيم
- ✓ خطوط كنتورية محفوظة

---

## 8. الملفات المعدلة

- ✅ `pipeline/preprocessing.py`
  - `classify_map_type()` - جديد
  - `denoise()` - محدث (ديناميكي)
  - `preprocess()` - محدث (يعيد map_type)
  - `preprocess_with_crop()` - محدث (يعيد map_type)

- ✅ `pipeline/pipeline.py`
  - `AdaptiveQAThresholds` - جديد
  - `get_adaptive_qa_thresholds()` - جديد
  - `calculate_layer_coverage_stats()` - جديد
  - `calculate_max_layer_ratio()` - جديد
  - `calculate_confidence_score()` - جديد
  - `run_pipeline()` - محدث (يستخدم detected_map_type)

- ✅ `pipeline/cc_postprocess.py`
  - `apply_adaptive_min_area()` - جديد
  - `vectorize_mask()` - محدث (map_type parameter)
  - `mask_to_geodataframe()` - محدث (map_type parameter)

---

## 9. الخطوات التالية

لتحسين النظام أكثر في المستقبل:

1. **Fine-tuning حدود التشبع**: اختبر على عينات أكبر
2. **معايير QA إضافية**: صحة الطوبولوجيا، التوازن الحجمي
3. **تدريب نموذج تصنيف**: بدلاً من الحد الثابت (15%)
4. **تحسين min_area للطبقات**: معايير أفضل بناءً على البيانات الحقيقية

---

**آخر تحديث:** 2026-05-24  
**النسخة:** Adaptive Map Detection v1.0
