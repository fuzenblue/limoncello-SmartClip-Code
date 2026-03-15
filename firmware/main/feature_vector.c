

#include "feature_vector.h"
#include "flicker_fft.h"
#include <string.h>
#include <math.h>
#include <time.h>
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "FEATURE_VEC";



static float pop_photophobia     = 0.80f;
static float pop_phonophobia     = 0.75f;
static float pop_pressure_sens   = 0.28f;
static float pop_voc_sens        = 0.45f;


void feature_vector_init(void)
{
    
    ESP_LOGI(TAG, "Feature vector module initialised with population priors:");
    ESP_LOGI(TAG, "  Photophobia: %.2f  Phonophobia: %.2f", pop_photophobia, pop_phonophobia);
    ESP_LOGI(TAG, "  Pressure:    %.2f  VOC:         %.2f", pop_pressure_sens, pop_voc_sens);
}


void feature_vector_build(feature_vector_t *fv,
                          const pressure_features_t *pf,
                          const voc_features_t *vf,
                          float flicker_index, float flicker_freq,
                          int8_t audio_class, float audio_conf, float audio_db,
                          uint8_t motion_active)
{
    
    
    fv->timestamp_unix = (uint32_t)(esp_timer_get_time() / 1000000ULL);
    fv->motion_active = motion_active;

    
    fv->flicker_index = flicker_index;
    fv->flicker_freq_hz = flicker_freq;
    fv->flicker_alert = flicker_is_alert(flicker_index, flicker_freq) ? 1 : 0;

    
    fv->pressure_hpa       = pf->pressure_hpa;
    fv->pressure_ddt_1h    = pf->ddt_1h;
    fv->pressure_ddt_6h    = pf->ddt_6h;
    fv->pressure_zscore    = pf->zscore;
    fv->pressure_drop_alert = (uint8_t)pf->drop_alert;

    
    fv->voc_raw     = vf->voc_raw;
    fv->voc_zscore  = vf->zscore;
    fv->voc_spike   = (uint8_t)vf->spike;
    fv->humidity_pct = vf->humidity_pct;
    fv->temp_celsius = vf->temp_celsius;

    
    fv->audio_class      = (uint8_t)audio_class;
    fv->audio_confidence = audio_conf;
    fv->audio_db_mean    = audio_db;

    
    fv->prior_photophobia          = pop_photophobia;
    fv->prior_phonophobia          = pop_phonophobia;
    fv->prior_pressure_sensitivity = pop_pressure_sens;
    fv->prior_voc_sensitivity      = pop_voc_sens;

    
    fv->risk_score =
        (float)fv->flicker_alert         * pop_photophobia   * 0.25f +
        (float)fv->pressure_drop_alert   * pop_pressure_sens * 0.30f +
        (float)fv->voc_spike             * pop_voc_sens      * 0.25f +
        (audio_class == 2 ? 1.0f : 0.0f) * pop_phonophobia   * 0.20f;
}


void feature_vector_serialise(const feature_vector_t *fv,
                               uint8_t *buf, size_t *len)
{
    

    size_t pos = 0;

    
    #define WRITE_U32(val) do { \
        uint32_t v = (val); \
        memcpy(&buf[pos], &v, 4); pos += 4; \
    } while(0)

    #define WRITE_FLOAT(val) do { \
        float v = (val); \
        memcpy(&buf[pos], &v, 4); pos += 4; \
    } while(0)

    #define WRITE_U8(val) do { \
        buf[pos] = (uint8_t)(val); pos += 1; \
    } while(0)

    
    WRITE_U32(fv->timestamp_unix);
    WRITE_U8(fv->motion_active);

    WRITE_FLOAT(fv->flicker_index);
    WRITE_FLOAT(fv->flicker_freq_hz);
    WRITE_U8(fv->flicker_alert);

    WRITE_FLOAT(fv->pressure_hpa);
    WRITE_FLOAT(fv->pressure_ddt_1h);
    WRITE_FLOAT(fv->pressure_ddt_6h);
    WRITE_FLOAT(fv->pressure_zscore);
    WRITE_U8(fv->pressure_drop_alert);

    WRITE_FLOAT(fv->voc_raw);
    WRITE_FLOAT(fv->voc_zscore);
    WRITE_U8(fv->voc_spike);
    WRITE_FLOAT(fv->humidity_pct);
    WRITE_FLOAT(fv->temp_celsius);

    WRITE_U8(fv->audio_class);
    WRITE_FLOAT(fv->audio_confidence);
    WRITE_FLOAT(fv->audio_db_mean);

    WRITE_FLOAT(fv->prior_photophobia);
    WRITE_FLOAT(fv->prior_phonophobia);
    WRITE_FLOAT(fv->prior_pressure_sensitivity);
    WRITE_FLOAT(fv->prior_voc_sensitivity);

    WRITE_FLOAT(fv->risk_score);

    *len = pos;

    #undef WRITE_U32
    #undef WRITE_FLOAT
    #undef WRITE_U8
}


void feature_vector_log(const feature_vector_t *fv)
{
    ESP_LOGI(TAG, "─── Feature Vector ───");
    ESP_LOGI(TAG, "  Motion: %s", fv->motion_active ? "ACTIVE" : "STATIONARY");
    ESP_LOGI(TAG, "  Light:  FI=%.4f  Freq=%.1fHz  Alert=%d",
             fv->flicker_index, fv->flicker_freq_hz, fv->flicker_alert);
    ESP_LOGI(TAG, "  Press:  %.1f hPa  ddt1h=%.3f  ddt6h=%.3f  z=%.2f  alert=%d",
             fv->pressure_hpa, fv->pressure_ddt_1h, fv->pressure_ddt_6h,
             fv->pressure_zscore, fv->pressure_drop_alert);
    ESP_LOGI(TAG, "  VOC:    %.0f Ω  z=%.2f  spike=%d  H=%.0f%%  T=%.1f°C",
             fv->voc_raw, fv->voc_zscore, fv->voc_spike,
             fv->humidity_pct, fv->temp_celsius);
    ESP_LOGI(TAG, "  Audio:  class=%d  conf=%.2f  dB=%.1f",
             fv->audio_class, fv->audio_confidence, fv->audio_db_mean);
    ESP_LOGI(TAG, "  Risk:   %.4f", fv->risk_score);
    ESP_LOGI(TAG, "──────────────────────");
}
