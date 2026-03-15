
#ifndef FEATURE_VECTOR_H
#define FEATURE_VECTOR_H

#include <stdint.h>
#include <stddef.h>
#include "pressure_voc_calc.h"


#define FEATURE_VECTOR_SIZE  27


typedef struct {
    
    uint32_t    timestamp_unix;          
    uint8_t     motion_active;           

    
    float       flicker_index;           
    float       flicker_freq_hz;         
    uint8_t     flicker_alert;           

    
    float       pressure_hpa;            
    float       pressure_ddt_1h;         
    float       pressure_ddt_6h;         
    float       pressure_zscore;         
    uint8_t     pressure_drop_alert;     

    
    float       voc_raw;                 
    float       voc_zscore;              
    uint8_t     voc_spike;               
    float       humidity_pct;            
    float       temp_celsius;            

    
    uint8_t     audio_class;             
    float       audio_confidence;        
    float       audio_db_mean;           

    
    float       prior_photophobia;       
    float       prior_phonophobia;
    float       prior_pressure_sensitivity;
    float       prior_voc_sensitivity;

    
    float       risk_score;              
} feature_vector_t;


void feature_vector_init(void);


void feature_vector_build(feature_vector_t *fv,
                          const pressure_features_t *pf,
                          const voc_features_t *vf,
                          float flicker_index, float flicker_freq,
                          int8_t audio_class, float audio_conf, float audio_db,
                          uint8_t motion_active);


void feature_vector_serialise(const feature_vector_t *fv,
                               uint8_t *buf, size_t *len);


void feature_vector_log(const feature_vector_t *fv);

#endif 
