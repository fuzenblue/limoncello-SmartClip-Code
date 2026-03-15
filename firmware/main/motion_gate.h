#ifndef MOTION_GATE_H
#define MOTION_GATE_H

#include <stdint.h>
#include <stdbool.h>

#define STATIONARY_THRESHOLD_SEC    300   
#define MOTION_THRESHOLD_G          0.08f  
#define GRAVITY_NOMINAL             1.0f   

void motion_gate_task(void *pvParameters);
bool motion_gate_is_active(void);

#endif 
