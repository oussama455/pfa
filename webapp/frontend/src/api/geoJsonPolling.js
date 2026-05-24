/**
 * webapp/frontend/src/api/geoJsonPolling.js
 *
 * Polling بـ Exponential Backoff لتجنب تكرار الطلبات المفرط.
 * عند الطلب: تبدأ بـ تأخير 1 ثانية، تزيد تصاعدياً حتى 10 ثوانٍ.
 * تتوقف عند الحصول على 200 (البيانات جاهزة) أو عند حدوث خطأ.
 */

/**
 * جلب GeoJSON مع Polling ذكي بـ Exponential Backoff
 * @param {number} mapId - معرف الخريطة
 * @returns {Promise} - تُرجع البيانات عند جاهزيتها
 */
export async function fetchGeoJsonWithPolling(mapId) {
    let delay = 1000;      // البداية بـ 1 ثانية
    const maxDelay = 10000; // أقصى حد: 10 ثوانٍ

    while (true) {
        try {
            const response = await fetch(`/api/maps/${mapId}/geojson/`);

            // الحالة 200: الملف جاهز تماماً ✅
            if (response.status === 200) {
                const geojsonData = await response.json();
                console.log(`✅ تم تحميل GeoJSON للخريطة ${mapId} بنجاح`);
                return geojsonData;
            }

            // الحالة 202: الخلفية لا تزال تعمل ⏳
            if (response.status === 202) {
                console.log(
                    `⏳ الخريطة ${mapId} قيد المعالجة... ` +
                    `سنحاول مجدداً بعد ${delay}ms`
                );
                
                // انتظار بـ Promise
                await new Promise(resolve => setTimeout(resolve, delay));
                
                // زيادة التأخير تصاعدياً (1.5x)
                delay = Math.min(delay * 1.5, maxDelay);
                continue;
            }

            // حالات أخرى: خطأ
            const errorText = await response.text();
            console.error(
                `❌ خطأ من الخادم (${response.status}) للخريطة ${mapId}: `,
                errorText
            );
            throw new Error(`HTTP ${response.status}: ${errorText}`);

        } catch (error) {
            console.error(
                `❌ خطأ في الاتصال بالخادم للخريطة ${mapId}:`,
                error.message
            );
            throw error;
        }
    }
}

/**
 * بديل بسيط للـ Polling بدون exponential backoff
 * (استخدم هذا إذا كنت تريد تأخير ثابت)
 * @param {number} mapId
 * @param {number} intervalMs - تأخير ثابت بـ ميلي ثانية (default: 2000)
 * @param {number} maxAttempts - أقصى عدد محاولات (default: 30)
 */
export async function fetchGeoJsonWithFixedPolling(
    mapId,
    intervalMs = 2000,
    maxAttempts = 30
) {
    let attempts = 0;

    while (attempts < maxAttempts) {
        try {
            const response = await fetch(`/api/maps/${mapId}/geojson/`);

            if (response.status === 200) {
                const geojsonData = await response.json();
                console.log(`✅ تم تحميل GeoJSON للخريطة ${mapId} بنجاح`);
                return geojsonData;
            }

            if (response.status === 202) {
                attempts++;
                console.log(
                    `⏳ المحاولة ${attempts}/${maxAttempts} للخريطة ${mapId}...`
                );
                await new Promise(resolve => setTimeout(resolve, intervalMs));
                continue;
            }

            throw new Error(`HTTP ${response.status}`);

        } catch (error) {
            attempts++;
            if (attempts >= maxAttempts) {
                console.error(
                    `❌ انتهت محاولات جلب GeoJSON للخريطة ${mapId}`
                );
                throw error;
            }
        }
    }
}
