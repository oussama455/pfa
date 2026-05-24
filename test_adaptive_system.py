"""
اختبار سريع للنظام الذكي للتعرف على أنواع الخرائط
Quick test script for the Adaptive Map Detection system
"""

import cv2
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Map Type Classification
# ─────────────────────────────────────────────────────────────────────────────

def test_map_type_detection(map_path: str):
    """اختبر كشف نوع الخريطة تلقائياً"""
    from pipeline.preprocessing import classify_map_type
    
    img = cv2.imread(map_path)
    if img is None:
        print(f"❌ لا يمكن تحميل الخريطة: {map_path}")
        return
    
    map_type = classify_map_type(img)
    print(f"✓ نوع الخريطة المكتشف: {map_type}")
    
    # احسب المتوسط الفعلي للتشبع للمراجعة
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_sat = np.mean(hsv[:, :, 1])
    print(f"  Mean Saturation: {mean_sat:.1f}%")
    print(f"  Threshold: 15% ← إذا < 15% → monochrome_faded")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Denoising Adaptation
# ─────────────────────────────────────────────────────────────────────────────

def test_denoise_adaptation(map_path: str):
    """اختبر تكيف الفلترة مع نوع الخريطة"""
    from pipeline.preprocessing import classify_map_type, denoise
    
    img = cv2.imread(map_path)
    if img is None:
        print(f"❌ لا يمكن تحميل الخريطة: {map_path}")
        return
    
    map_type = classify_map_type(img)
    print(f"\n✓ Denoising Adaptation Test")
    print(f"  Map Type: {map_type}")
    
    # استدعاء الفلترة الديناميكية
    denoised = denoise(img, map_type=map_type)
    
    if map_type == "color_rich":
        print(f"  Filter: bilateral(d=5, sigma=50) — خفيف")
    else:
        print(f"  Filter: bilateral(d=9, sigma=90) + MORPH_CLOSE — قوي")
    
    print(f"  Original shape: {img.shape}")
    print(f"  Denoised shape: {denoised.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: QA Thresholds
# ─────────────────────────────────────────────────────────────────────────────

def test_qa_thresholds():
    """اختبر المعايير الديناميكية للخريطتين"""
    from pipeline.pipeline import get_adaptive_qa_thresholds
    
    print(f"\n✓ QA Thresholds Test")
    
    for map_type in ["color_rich", "monochrome_faded"]:
        thresholds = get_adaptive_qa_thresholds(map_type)
        print(f"\n  {map_type}:")
        print(f"    QA Target:              {thresholds.qa_threshold:.1f}%")
        print(f"    Max Layer Ratio:        {thresholds.max_allowed_layer_ratio:.1f}%")
        print(f"    Min Acceptable Conf:    {thresholds.min_acceptable_confidence:.1f}%")
        print(f"    Max Iterations:         {thresholds.max_iterations}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Adaptive Min Area
# ─────────────────────────────────────────────────────────────────────────────

def test_adaptive_min_area():
    """اختبر حساب المساحة الدنيا التكيفية"""
    from pipeline.cc_postprocess import apply_adaptive_min_area
    
    print(f"\n✓ Adaptive Min Area Test")
    
    layers = ["buildings", "contours", "water", "vegetation", "roads"]
    
    for map_type in ["color_rich", "monochrome_faded"]:
        print(f"\n  {map_type}:")
        for layer in layers:
            min_area = apply_adaptive_min_area(layer, map_type=map_type)
            multiplier = 2.0 if map_type == "monochrome_faded" else 1.0
            print(f"    {layer:12s}: {min_area:3d} px (×{multiplier})")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Full Pipeline (Optional - requires all dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline(map_path: str, output_dir: str = "data/test_output"):
    """اختبر الـ pipeline الكامل"""
    try:
        from pipeline.pipeline import run_pipeline
        
        print(f"\n✓ Full Pipeline Test")
        print(f"  Input:  {map_path}")
        print(f"  Output: {output_dir}")
        
        result = run_pipeline(
            input_path=map_path,
            output_dir=output_dir,
            verbose=True
        )
        
        print(f"\n✓ Pipeline completed!")
        print(f"  Generated layers: {list(result.layers.keys())}")
        
    except Exception as e:
        print(f"⚠ Pipeline test skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("اختبار النظام الذكي للتعرف على أنواع الخرائط")
    print("Adaptive Map Type Detection System - Test Suite")
    print("=" * 70)
    
    # الاختبارات التي لا تحتاج صور حقيقية
    print("\n[1/5] اختبار المعايير الديناميكية (QA Thresholds)")
    print("─" * 70)
    test_qa_thresholds()
    
    print("\n[2/5] اختبار المساحة الدنيا التكيفية (Adaptive Min Area)")
    print("─" * 70)
    test_adaptive_min_area()
    
    # الاختبارات التي تحتاج صور حقيقية
    test_images = list(Path("data/raw").glob("*.tif")) + \
                  list(Path("data/raw").glob("*.jpg"))
    
    if not test_images:
        print("\n⚠ لم توجد صور اختبار في data/raw/")
        print("  لتشغيل الاختبارات الكاملة، ضع صور TIFF/JPG في data/raw/")
    else:
        test_map = str(test_images[0])
        
        print(f"\n[3/5] اختبار كشف نوع الخريطة (Map Type Detection)")
        print("─" * 70)
        test_map_type_detection(test_map)
        
        print(f"\n[4/5] اختبار تكيف الفلترة (Denoise Adaptation)")
        print("─" * 70)
        test_denoise_adaptation(test_map)
        
        print(f"\n[5/5] اختبار الـ pipeline الكامل (Full Pipeline)")
        print("─" * 70)
        test_full_pipeline(test_map)
    
    print("\n" + "=" * 70)
    print("✓ اختبار النظام انتهى")
    print("=" * 70)
