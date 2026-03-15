#ifndef FLICKER_FFT_H
#define FLICKER_FFT_H

#include <stdint.h>
#include <stdbool.h>

#define FFT_SAMPLE_RATE       800     
#define FFT_WINDOW_SIZE       512     
#define FFT_FLICKER_BAND_LOW  90      
#define FFT_FLICKER_BAND_HIGH 210     
#define FFT_FLICKER_THRESHOLD 0.08f   

void flicker_fft_init(void);

void flicker_compute(const uint16_t *adc_buf, float *fi_out, float *freq_out);

void flicker_get_latest(float *fi_out, float *freq_out);

bool flicker_is_alert(float flicker_index, float dominant_freq);

#endif 
