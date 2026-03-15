#include "motion_gate.h"
#include "sensor_grid.h"
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "MOTION_GATE";

static volatile bool device_active = true;  
static int64_t last_motion_time_us = 0;     

void motion_gate_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Motion gate task started (threshold=%.2fg, timeout=%ds)",
             MOTION_THRESHOLD_G, STATIONARY_THRESHOLD_SEC);

    last_motion_time_us = esp_timer_get_time();  

    imu_accel_t accel;

    while (1) {
        esp_err_t ret = imu_read_accel(&accel);
        if (ret == ESP_OK) {
            float deviation = fabsf(accel.mag - GRAVITY_NOMINAL);

            if (deviation > MOTION_THRESHOLD_G) {
                
                last_motion_time_us = esp_timer_get_time();

                if (!device_active) {
                    
                    device_active = true;
                    ESP_LOGI(TAG, "Motion detected — switching to ACTIVE mode "
                             "(ax=%.3f ay=%.3f az=%.3f mag=%.3f)",
                             accel.ax, accel.ay, accel.az, accel.mag);
                }
            }
        }

        int64_t now_us = esp_timer_get_time();
        int64_t elapsed_us = now_us - last_motion_time_us;
        int64_t threshold_us = (int64_t)STATIONARY_THRESHOLD_SEC * 1000000LL;

        if (elapsed_us > threshold_us && device_active) {
            
            device_active = false;
            ESP_LOGI(TAG, "No motion for %d seconds — switching to STATIONARY mode",
                     STATIONARY_THRESHOLD_SEC);
        }

        TickType_t delay = device_active
                           ? pdMS_TO_TICKS(100)
                           : pdMS_TO_TICKS(500);
        vTaskDelay(delay);
    }
}


bool motion_gate_is_active(void)
{
    return device_active;
}
