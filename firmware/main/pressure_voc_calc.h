#ifndef PRESSURE_VOC_CALC_H
#define PRESSURE_VOC_CALC_H

#include <stdint.h>
#include <stdbool.h>


#define PRESSURE_RING_SIZE      4320
#define PRESSURE_STEPS_1H       6       
#define PRESSURE_STEPS_6H      36       

#define VOC_RING_SIZE           1008
#define VOC_SPIKE_THRESHOLD     2.0f    

typedef struct {
    float pressure_hpa;          
    float ddt_1h;                
    float ddt_6h;                
    float std_6h;                
    float zscore;                
    int   drop_alert_1h;         
    int   drop_alert_6h;         
    int   trigger;               
} pressure_features_t;

typedef struct {
    float voc_raw;               
    float zscore;                
    float ddt_10min;             
    int   spike;                 
    int   persistent_spike;      
    float humidity_pct;          
    float temp_celsius;          
} voc_features_t;

void pressure_calc_init(void);
void pressure_update(float pressure_hpa);
void pressure_get_features(pressure_features_t *out);
void voc_update(float gas_resistance_ohm);
void voc_get_features(voc_features_t *out);

#endif 
