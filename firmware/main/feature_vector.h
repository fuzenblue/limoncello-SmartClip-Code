
#ifndef FEATURE_VECTOR_H
#define FEATURE_VECTOR_H

#include <stdint.h>
#include <stddef.h>
#include "pressure_voc_calc.h"


#define FEATURE_VECTOR_SIZE  27


typedef struct {
    /* Metadata */
    uint32_t    timestamp_unix;          
    uint8_t     motion_active;           

    /* Light features (3) */
    float       flicker_index;           
    float       flicker_freq_hz;         
    uint8_t     flicker_alert;           

    /* Pressure features (8) */
    float       pressure_hpa;            
    float       pressure_ddt_1h;         
    float       pressure_ddt_6h;         
    float       pressure_std_6h;         
    float       pressure_zscore;         
    uint8_t     pressure_drop_alert_1h;  
    uint8_t     pressure_drop_alert_6h;  
    uint8_t     pressure_trigger;        

    /* VOC features (7) */
    float       voc_raw;                 
    float       voc_zscore;              
    float       voc_ddt_10min;           
    uint8_t     voc_spike;               
    uint8_t     voc_persistent_spike;    
    float       humidity_pct;            
    float       temp_celsius;            

    /* Audio features (3) */
    uint8_t     audio_class;             
    float       audio_confidence;        
    float       audio_db_mean;           

    /* User priors (4) */
    float       prior_photophobia;       
    float       prior_phonophobia;
    float       prior_pressure_sensitivity;
    float       prior_voc_sensitivity;

    /* Risk score (1) */
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
