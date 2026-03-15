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

    
    float past_1h;
    if (ring_stat_read_back(&pressure_ring, PRESSURE_STEPS_1H, &past_1h)) {
        out->ddt_1h = current - past_1h;  
    } else {
        out->ddt_1h = 0.0f;  
    }

    
    float past_6h;
    if (ring_stat_read_back(&pressure_ring, PRESSURE_STEPS_6H, &past_6h)) {
        out->ddt_6h = (current - past_6h) / 6.0f;  
    } else {
        out->ddt_6h = 0.0f;
    }

    float mean = ring_stat_mean(&pressure_ring);
    float std  = ring_stat_std(&pressure_ring);
    out->zscore = (current - mean) / (std + EPSILON);

    out->drop_alert = (out->ddt_1h < -1.5f) ? 1 : 0;

    ESP_LOGD(TAG, "Pressure features: P=%.2f ddt1h=%.3f ddt6h=%.3f z=%.2f alert=%d",
             out->pressure_hpa, out->ddt_1h, out->ddt_6h,
             out->zscore, out->drop_alert);
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

    float mean = ring_stat_mean(&voc_ring);
    float std  = ring_stat_std(&voc_ring);
    out->zscore = -((current - mean) / (std + EPSILON));
    out->spike = (out->zscore > VOC_SPIKE_THRESHOLD) ? 1 : 0;

    ESP_LOGD(TAG, "VOC features: raw=%.0f z=%.2f spike=%d",
             out->voc_raw, out->zscore, out->spike);
}
