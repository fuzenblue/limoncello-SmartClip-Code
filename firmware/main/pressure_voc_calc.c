#include "pressure_voc_calc.h"
#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char *TAG = "PRESS_VOC_CALC";
#define EPSILON  1e-6f

typedef struct {
    float  *data;          
    int     capacity;       
    int     head;           
    int     count;          
    double  welford_mean;   
    double  welford_m2;     
} ring_stat_t;

static void ring_stat_init(ring_stat_t *rs, float *buffer, int capacity)
{
    rs->data = buffer;
    rs->capacity = capacity;
    rs->head = 0;
    rs->count = 0;
    rs->welford_mean = 0.0;
    rs->welford_m2 = 0.0;
    memset(buffer, 0, capacity * sizeof(float));
}


static void ring_stat_push(ring_stat_t *rs, float val)
{
    if (rs->count >= rs->capacity) {
        float old_val = rs->data[rs->head];
        double delta_old = (double)old_val - rs->welford_mean;
        rs->count--;
        if (rs->count > 0) {
            rs->welford_mean -= delta_old / (double)rs->count;
            double delta2_old = (double)old_val - rs->welford_mean;
            rs->welford_m2 -= delta_old * delta2_old;
        } else {
            rs->welford_mean = 0.0;
            rs->welford_m2 = 0.0;
        }
    }

    rs->data[rs->head] = val;
    rs->head = (rs->head + 1) % rs->capacity;
    rs->count++;
    double delta = (double)val - rs->welford_mean;
    rs->welford_mean += delta / (double)rs->count;
    double delta2 = (double)val - rs->welford_mean;
    rs->welford_m2 += delta * delta2;

    if (rs->welford_m2 < 0.0) rs->welford_m2 = 0.0;
}


static float ring_stat_mean(const ring_stat_t *rs)
{
    if (rs->count == 0) return 0.0f;
    return (float)rs->welford_mean;
}


static float ring_stat_std(const ring_stat_t *rs)
{
    if (rs->count < 2) return EPSILON;  
    return (float)sqrt(rs->welford_m2 / (double)rs->count);
}

static bool ring_stat_read_back(const ring_stat_t *rs, int offset, float *out)
{
    if (offset >= rs->count) return false;

    int idx = (rs->head - 1 - offset + rs->capacity) % rs->capacity;
    *out = rs->data[idx];
    return true;
}

static float pressure_buffer[PRESSURE_RING_SIZE];
static ring_stat_t pressure_ring;

static float voc_buffer[VOC_RING_SIZE];
static ring_stat_t voc_ring;


void pressure_calc_init(void)
{
    ring_stat_init(&pressure_ring, pressure_buffer, PRESSURE_RING_SIZE);
    ring_stat_init(&voc_ring, voc_buffer, VOC_RING_SIZE);
    ESP_LOGI(TAG, "Pressure/VOC calculators initialised "
             "(P_buf=%d, VOC_buf=%d)", PRESSURE_RING_SIZE, VOC_RING_SIZE);
}


void pressure_update(float pressure_hpa)
{
    ring_stat_push(&pressure_ring, pressure_hpa);
}


void pressure_get_features(pressure_features_t *out)
{
    float current;
    if (!ring_stat_read_back(&pressure_ring, 0, &current)) {
        memset(out, 0, sizeof(pressure_features_t));
        return;
    }
    out->pressure_hpa = current;

    /* dP/dt over 1 hour (6 steps) */
    float past_1h;
    if (ring_stat_read_back(&pressure_ring, PRESSURE_STEPS_1H, &past_1h)) {
        out->ddt_1h = current - past_1h;
    } else {
        out->ddt_1h = 0.0f;
    }

    /* dP/dt over 6 hours (36 steps) */
    float past_6h;
    if (ring_stat_read_back(&pressure_ring, PRESSURE_STEPS_6H, &past_6h)) {
        out->ddt_6h = (current - past_6h) / 6.0f;
    } else {
        out->ddt_6h = 0.0f;
    }

    /* Windowed standard deviation (6 hours = 36 steps) */
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = 0;
    for (int i = 0; i < PRESSURE_STEPS_6H; i++) {
        float val;
        if (ring_stat_read_back(&pressure_ring, i, &val)) {
            sum += val;
            sum_sq += val * val;
            count++;
        }
    }
    if (count > 1) {
        float window_mean = sum / (float)count;
        out->std_6h = sqrtf((sum_sq / (float)count) - (window_mean * window_mean));
    } else {
        out->std_6h = EPSILON;
    }

    /* 30-day z-score */
    float mean_30d = ring_stat_mean(&pressure_ring);
    float std_30d  = ring_stat_std(&pressure_ring);
    out->zscore = (current - mean_30d) / (std_30d + EPSILON);

    /* Alerts */
    out->drop_alert_1h = (out->ddt_1h < -1.5f) ? 1 : 0;
    out->drop_alert_6h = (out->ddt_6h < -0.5f) ? 1 : 0;
    out->trigger = (out->drop_alert_1h && out->drop_alert_6h) ? 1 : 0;

    ESP_LOGD(TAG, "Pressure: %.1f ddt1=%.2f ddt6=%.2f std6=%.2f z=%.2f alert1=%d alert6=%d trig=%d",
             out->pressure_hpa, out->ddt_1h, out->ddt_6h, out->std_6h,
             out->zscore, out->drop_alert_1h, out->drop_alert_6h, out->trigger);
}

void voc_update(float gas_resistance_ohm)
{
    ring_stat_push(&voc_ring, gas_resistance_ohm);
}


void voc_get_features(voc_features_t *out)
{
    float current;
    if (!ring_stat_read_back(&voc_ring, 0, &current)) {
        memset(out, 0, sizeof(voc_features_t));
        return;
    }
    out->voc_raw = current;

    /* dR/dt per 10 min */
    float prev;
    if (ring_stat_read_back(&voc_ring, 1, &prev)) {
        out->ddt_10min = (current - prev) / 10.0f;
    } else {
        out->ddt_10min = 0.0f;
    }

    /* 7-day z-score */
    float mean_7d = ring_stat_mean(&voc_ring);
    float std_7d  = ring_stat_std(&voc_ring);
    out->zscore = -((current - mean_7d) / (std_7d + EPSILON));

    /* Spike detection */
    out->spike = (out->zscore > VOC_SPIKE_THRESHOLD) ? 1 : 0;

    /* Persistent spike (3 consecutive windows) */
    int spike_count = 0;
    for (int i = 0; i < 3; i++) {
        float val;
        if (ring_stat_read_back(&voc_ring, i, &val)) {
            float z = -((val - mean_7d) / (std_7d + EPSILON));
            if (z > VOC_SPIKE_THRESHOLD) spike_count++;
        }
    }
    out->persistent_spike = (spike_count >= 3) ? 1 : 0;

    ESP_LOGD(TAG, "VOC: raw=%.0f ddt=%.1f z=%.2f spike=%d persist=%d",
             out->voc_raw, out->ddt_10min, out->zscore, out->spike, out->persistent_spike);
}
