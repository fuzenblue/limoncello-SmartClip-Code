

#include "audio_features.h"
#include "sensor_grid.h"
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_dsp.h"

static const char *TAG = "AUDIO_FEAT";





static int32_t audio_raw[AUDIO_FRAME_LEN];


static float audio_float[AUDIO_FRAME_LEN];


static float stft_buf[AUDIO_N_FFT * 2];


static float mel_energies[AUDIO_N_MELS];


static float mfcc_accum[AUDIO_N_MFCC];
static int   mfcc_frame_count;





static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}



static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}



static void apply_mel_filterbank(const float *power_spectrum,
                                  float *mel_out,
                                  int n_fft, int sample_rate)
{
    float f_min = 0.0f;
    float f_max = (float)sample_rate / 2.0f;   

    float mel_min = hz_to_mel(f_min);
    float mel_max = hz_to_mel(f_max);

    
    float mel_points[AUDIO_N_MELS + 2];
    for (int i = 0; i < AUDIO_N_MELS + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) *
                        (float)i / (float)(AUDIO_N_MELS + 1);
    }

    
    int bin_indices[AUDIO_N_MELS + 2];
    for (int i = 0; i < AUDIO_N_MELS + 2; i++) {
        float hz = mel_to_hz(mel_points[i]);
        bin_indices[i] = (int)roundf(hz * (float)n_fft / (float)sample_rate);
        if (bin_indices[i] > n_fft / 2) bin_indices[i] = n_fft / 2;
    }

    
    int n_bins = n_fft / 2 + 1;
    for (int m = 0; m < AUDIO_N_MELS; m++) {
        int f_left   = bin_indices[m];
        int f_center = bin_indices[m + 1];
        int f_right  = bin_indices[m + 2];

        mel_out[m] = 0.0f;

        
        for (int k = f_left; k <= f_center && k < n_bins; k++) {
            float weight = 0.0f;
            if (f_center > f_left) {
                weight = (float)(k - f_left) / (float)(f_center - f_left);
            }
            mel_out[m] += power_spectrum[k] * weight;
        }

        
        for (int k = f_center + 1; k <= f_right && k < n_bins; k++) {
            float weight = 0.0f;
            if (f_right > f_center) {
                weight = (float)(f_right - k) / (float)(f_right - f_center);
            }
            mel_out[m] += power_spectrum[k] * weight;
        }

        
        if (mel_out[m] < 1e-10f) mel_out[m] = 1e-10f;
    }
}





static void apply_dct(const float *mel_log, float *dct_out)
{
    for (int k = 0; k < AUDIO_N_MFCC; k++) {
        float sum = 0.0f;
        for (int n = 0; n < AUDIO_N_MELS; n++) {
            
            float angle = (float)M_PI * (float)k *
                         (2.0f * (float)n + 1.0f) / (2.0f * (float)AUDIO_N_MELS);
            sum += mel_log[n] * cosf(angle);
        }
        dct_out[k] = sum;
    }
}





static float compute_db_spl(const float *audio, int len)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < len; i++) {
        sum_sq += audio[i] * audio[i];
    }
    float rms = sqrtf(sum_sq / (float)len);

    if (rms < 1e-8f) return 20.0f;  

    
    float db_fs = 20.0f * log10f(rms);
    float db_spl = db_fs + 120.0f;

    
    if (db_spl < 20.0f) db_spl = 20.0f;
    if (db_spl > 130.0f) db_spl = 130.0f;

    return db_spl;
}





static void classify_mfcc(const float *mfcc, int8_t *class_out,
                           float *confidence_out)
{
    
    float energy = mfcc[0];          
    float slope = mfcc[1];           

    
    if (energy > -120.0f && fabsf(slope) > 80.0f) {
        
        *class_out = AUDIO_CLASS_TRIGGER;
        *confidence_out = 0.75f;
    } else if (energy > -180.0f) {
        
        *class_out = AUDIO_CLASS_TRAFFIC;
        *confidence_out = 0.70f;
    } else {
        
        *class_out = AUDIO_CLASS_QUIET;
        *confidence_out = 0.85f;
    }
}




void audio_task_run(int8_t *class_out, float *confidence_out, float *db_out)
{
    ESP_LOGI(TAG, "Recording 1s audio from INMP441...");

    
    size_t samples_read = inmp441_read_frame(audio_raw, AUDIO_FRAME_LEN);
    if (samples_read < AUDIO_FRAME_LEN / 2) {
        ESP_LOGW(TAG, "Insufficient audio samples (%zu/%d)", samples_read,
                 AUDIO_FRAME_LEN);
        *class_out = AUDIO_CLASS_QUIET;
        *confidence_out = 0.5f;
        *db_out = 30.0f;
        return;
    }

    
    float max_val = 8388608.0f;  
    for (size_t i = 0; i < samples_read; i++) {
        audio_float[i] = (float)audio_raw[i] / max_val;
    }

    
    *db_out = compute_db_spl(audio_float, (int)samples_read);

    

    
    memset(mfcc_accum, 0, sizeof(mfcc_accum));
    mfcc_frame_count = 0;

    
    float hamming[AUDIO_N_FFT];
    for (int n = 0; n < AUDIO_N_FFT; n++) {
        hamming[n] = 0.54f - 0.46f * cosf(2.0f * M_PI * (float)n /
                                            (float)(AUDIO_N_FFT - 1));
    }

    
    float power[AUDIO_N_FFT / 2 + 1];
    float log_mel[AUDIO_N_MELS];
    float frame_mfcc[AUDIO_N_MFCC];

    
    int total_frames = 0;
    for (int start = 0; start + AUDIO_N_FFT <= (int)samples_read; start += AUDIO_HOP_LEN) {
        
        memset(stft_buf, 0, sizeof(stft_buf));

        
        for (int i = 0; i < AUDIO_N_FFT; i++) {
            stft_buf[i * 2]     = audio_float[start + i] * hamming[i];  
            stft_buf[i * 2 + 1] = 0.0f;                                
        }

        
        dsps_fft2r_fc32(stft_buf, AUDIO_N_FFT);
        dsps_bit_rev_fc32(stft_buf, AUDIO_N_FFT);

        
        for (int k = 0; k <= AUDIO_N_FFT / 2; k++) {
            float re = stft_buf[k * 2];
            float im = stft_buf[k * 2 + 1];
            power[k] = re * re + im * im;
        }

        
        apply_mel_filterbank(power, mel_energies, AUDIO_N_FFT, AUDIO_SAMPLE_RATE);

        
        for (int m = 0; m < AUDIO_N_MELS; m++) {
            log_mel[m] = logf(mel_energies[m]);
        }

        
        apply_dct(log_mel, frame_mfcc);

        
        for (int i = 0; i < AUDIO_N_MFCC; i++) {
            mfcc_accum[i] += frame_mfcc[i];
        }
        total_frames++;
    }

    
    if (total_frames > 0) {
        for (int i = 0; i < AUDIO_N_MFCC; i++) {
            mfcc_accum[i] /= (float)total_frames;
        }
    }
    mfcc_frame_count = total_frames;

    ESP_LOGI(TAG, "MFCC extraction: %d frames, MFCC[0]=%.2f MFCC[1]=%.2f",
             total_frames, mfcc_accum[0], mfcc_accum[1]);

    
    classify_mfcc(mfcc_accum, class_out, confidence_out);

    ESP_LOGI(TAG, "Audio classification: class=%d conf=%.2f dB=%.1f",
             *class_out, *confidence_out, *db_out);
}
