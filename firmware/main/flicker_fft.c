

#include "flicker_fft.h"
#include "sensor_grid.h"
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_dsp.h"   

static const char *TAG = "FLICKER_FFT";



static float hamming_window[FFT_WINDOW_SIZE];


static float fft_input[FFT_WINDOW_SIZE * 2];   
static float power_spectrum[FFT_WINDOW_SIZE / 2];


static volatile float latest_flicker_index = 0.0f;
static volatile float latest_dominant_freq = 0.0f;



void flicker_fft_init(void)
{
    
    for (int n = 0; n < FFT_WINDOW_SIZE; n++) {
        
        hamming_window[n] = 0.54f - 0.46f * cosf(2.0f * M_PI * (float)n /
                                                   (float)(FFT_WINDOW_SIZE - 1));
    }

    
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, FFT_WINDOW_SIZE);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FFT init failed: %s", esp_err_to_name(ret));
    }

    ESP_LOGI(TAG, "Flicker FFT initialised (window=%d, band=%d-%dHz)",
             FFT_WINDOW_SIZE, FFT_FLICKER_BAND_LOW, FFT_FLICKER_BAND_HIGH);
}


void flicker_compute(const uint16_t *adc_buf, float *fi_out, float *freq_out)
{
    
    for (int i = 0; i < FFT_WINDOW_SIZE; i++) {
        float normalised = ((float)adc_buf[i] - 2048.0f) / 2048.0f;

        
        fft_input[i * 2]     = normalised * hamming_window[i];  
        fft_input[i * 2 + 1] = 0.0f;                           
    }

    
    dsps_fft2r_fc32(fft_input, FFT_WINDOW_SIZE);

    
    dsps_bit_rev_fc32(fft_input, FFT_WINDOW_SIZE);

    
    float freq_resolution = (float)FFT_SAMPLE_RATE / (float)FFT_WINDOW_SIZE;

    for (int i = 0; i < FFT_WINDOW_SIZE / 2; i++) {
        float re = fft_input[i * 2];
        float im = fft_input[i * 2 + 1];
        power_spectrum[i] = re * re + im * im;
    }

    
    int band_low_idx  = (int)ceilf((float)FFT_FLICKER_BAND_LOW / freq_resolution);
    int band_high_idx = (int)floorf((float)FFT_FLICKER_BAND_HIGH / freq_resolution);

    
    if (band_low_idx < 1) band_low_idx = 1;
    if (band_high_idx >= FFT_WINDOW_SIZE / 2) band_high_idx = FFT_WINDOW_SIZE / 2 - 1;

    float band_power = 0.0f;
    float total_power = 0.0f;
    int dominant_bin = band_low_idx;
    float max_band_power = 0.0f;

    
    for (int i = 1; i < FFT_WINDOW_SIZE / 2; i++) {
        total_power += power_spectrum[i];
    }

    
    for (int i = band_low_idx; i <= band_high_idx; i++) {
        band_power += power_spectrum[i];
        if (power_spectrum[i] > max_band_power) {
            max_band_power = power_spectrum[i];
            dominant_bin = i;
        }
    }

    
    float flicker_index;
    float dominant_freq;

    if (total_power > 1e-12f) {
        flicker_index = band_power / total_power;
    } else {
        flicker_index = 0.0f;  
    }

    dominant_freq = (float)dominant_bin * freq_resolution;

    
    *fi_out = flicker_index;
    *freq_out = dominant_freq;

    
    latest_flicker_index = flicker_index;
    latest_dominant_freq = dominant_freq;
}


void flicker_get_latest(float *fi_out, float *freq_out)
{
    *fi_out = latest_flicker_index;
    *freq_out = latest_dominant_freq;
}


bool flicker_is_alert(float flicker_index, float dominant_freq)
{
    
    return (flicker_index > FFT_FLICKER_THRESHOLD) &&
           (dominant_freq >= (float)FFT_FLICKER_BAND_LOW) &&
           (dominant_freq <= (float)FFT_FLICKER_BAND_HIGH);
}
