
#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

#include <stdint.h>


#define AUDIO_SAMPLE_RATE   16000    
#define AUDIO_FRAME_LEN     16000    
#define AUDIO_N_MFCC        13       
#define AUDIO_N_FFT         512      
#define AUDIO_HOP_LEN       160      
#define AUDIO_N_MELS        40       


#define AUDIO_CLASS_QUIET    0        
#define AUDIO_CLASS_TRAFFIC  1        
#define AUDIO_CLASS_TRIGGER  2        


void audio_task_run(int8_t *class_out, float *confidence_out, float *db_out);

#endif 
