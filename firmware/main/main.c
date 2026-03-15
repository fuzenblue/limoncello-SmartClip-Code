#include <stdio.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_err.h"
#include "nvs_flash.h"           
#include "esp_timer.h"            

#include "driver/i2c.h"          
#include "driver/i2s_std.h"      
#include "driver/adc.h"          
#include "esp_adc/adc_oneshot.h" 

#include "sensor_grid.h"
#include "flicker_fft.h"
#include "pressure_voc_calc.h"
#include "audio_features.h"
#include "feature_vector.h"
#include "ble_transmit.h"
#include "motion_gate.h"

#include "model_data.h"

static const char *TAG = "SMARTCLIP_MAIN";

#define SENSOR_TASK_STACK_SIZE     (8192)     
#define PHOTODIODE_TASK_STACK_SIZE (4096)     
#define AUDIO_TASK_STACK_SIZE      (16384)    
#define MOTION_TASK_STACK_SIZE     (4096)     
#define BLE_TASK_STACK_SIZE        (4096)     

#define SENSOR_TASK_PRIORITY       (5)
#define PHOTODIODE_TASK_PRIORITY   (7)   
#define AUDIO_TASK_PRIORITY        (6)
#define MOTION_TASK_PRIORITY       (4)
#define BLE_TASK_PRIORITY          (3)

#define CORE_0                     (0)   
#define CORE_1                     (1)   

#define SENSOR_INTERVAL_MS         (600000)

static TaskHandle_t sensor_task_handle    = NULL;
static TaskHandle_t photodiode_task_handle = NULL;
static TaskHandle_t audio_task_handle     = NULL;
static TaskHandle_t motion_task_handle    = NULL;
static TaskHandle_t ble_task_handle       = NULL;

static void sensor_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Sensor task started (interval: %d ms)", SENSOR_INTERVAL_MS);

    TickType_t last_wake_time = xTaskGetTickCount();

    bme680_data_t     env_data;           
    float             flicker_index;       
    float             flicker_freq;        
    int8_t            audio_class;         
    float             audio_confidence;    
    float             audio_db;            
    pressure_features_t pressure_feat;     
    voc_features_t      voc_feat;          
    feature_vector_t    fv;                

    while (1) {
        vTaskDelayUntil(&last_wake_time,
                        pdMS_TO_TICKS(SENSOR_INTERVAL_MS));

        ESP_LOGI(TAG, "=== Sensor window triggered ===");

        if (!motion_gate_is_active()) {
            ESP_LOGI(TAG, "Device stationary — reduced sensing mode");
        }

        esp_err_t ret = bme680_read(&env_data);
        if (ret == ESP_OK) {
            ESP_LOGI(TAG, "BME680: P=%.1f hPa  T=%.1f°C  H=%.0f%%  Gas=%.0f Ω",
                     env_data.pressure_hpa,
                     env_data.temp_celsius,
                     env_data.humidity_pct,
                     env_data.gas_resistance_ohm);

            pressure_update(env_data.pressure_hpa);
            pressure_get_features(&pressure_feat);

            voc_update(env_data.gas_resistance_ohm);
            voc_get_features(&voc_feat);
            voc_feat.humidity_pct = env_data.humidity_pct;
            voc_feat.temp_celsius = env_data.temp_celsius;
        } else {
            ESP_LOGW(TAG, "BME680 read failed: %s", esp_err_to_name(ret));
        }

        flicker_get_latest(&flicker_index, &flicker_freq);
        ESP_LOGI(TAG, "Flicker: FI=%.4f  Freq=%.1f Hz  Alert=%s",
                 flicker_index, flicker_freq,
                 flicker_is_alert(flicker_index, flicker_freq) ? "YES" : "no");

        if (motion_gate_is_active()) {
            audio_task_run(&audio_class, &audio_confidence, &audio_db);
            ESP_LOGI(TAG, "Audio: class=%d  conf=%.2f  dB=%.1f",
                     audio_class, audio_confidence, audio_db);
        } else {
            audio_class = 0;        
            audio_confidence = 0.9f;
            audio_db = 35.0f;
        }

        feature_vector_build(&fv, &pressure_feat, &voc_feat,
                             flicker_index, flicker_freq,
                             audio_class, audio_confidence, audio_db,
                             motion_gate_is_active());
        feature_vector_log(&fv);

        if (ble_is_connected()) {
            uint8_t buf[sizeof(feature_vector_t)];
            size_t len = 0;
            feature_vector_serialise(&fv, buf, &len);
            ble_transmit_feature_vector(buf, len);
            ESP_LOGI(TAG, "BLE: Transmitted %zu bytes", len);
        } else {
            ESP_LOGW(TAG, "BLE: No client connected — data not transmitted");
        }

        ESP_LOGI(TAG, "=== Sensor window complete ===\n");
    }
}


void app_main(void)
{
    ESP_LOGI(TAG, "╔══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   Smart Clip — Firmware v1.0.0       ║");
    ESP_LOGI(TAG, "║   Migraine Sensitivity Companion     ║");
    ESP_LOGI(TAG, "╚══════════════════════════════════════╝");

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        
        ESP_LOGW(TAG, "NVS: Erasing and reinitialising");
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    ESP_LOGI(TAG, "✓ NVS initialised");

    i2c_config_t i2c_conf = {
        .mode = I2C_MODE_MASTER,        
        .sda_io_num = GPIO_NUM_21,       
        .scl_io_num = GPIO_NUM_22,       
        .sda_pullup_en = GPIO_PULLUP_ENABLE,  
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = 400000,      
    };
    ESP_ERROR_CHECK(i2c_param_config(I2C_NUM_0, &i2c_conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_NUM_0, I2C_MODE_MASTER, 0, 0, 0));
    ESP_LOGI(TAG, "✓ I2C bus initialised (SDA=21, SCL=22, 400kHz)");

    inmp441_init();
    ESP_LOGI(TAG, "✓ I2S initialised for INMP441 (16kHz, 32-bit, mono)");

    photodiode_init();
    ESP_LOGI(TAG, "✓ ADC initialised for photodiode (800Hz timer)");
    bme680_init();
    ESP_LOGI(TAG, "✓ BME680 initialised");

    imu_init();
    ESP_LOGI(TAG, "✓ IMU (MPU-6050) initialised");

    flicker_fft_init();
    ESP_LOGI(TAG, "✓ Flicker FFT module initialised");

    pressure_calc_init();
    ESP_LOGI(TAG, "✓ Pressure calculator initialised");

    feature_vector_init();
    ESP_LOGI(TAG, "✓ Feature vector module initialised");

    ble_init();
    ESP_LOGI(TAG, "✓ BLE GATT server initialised and advertising");

    ESP_LOGI(TAG, "Loading AI model (%u bytes from flash)...",
             gru_model_data_len);
    if (gru_model_data_len > 0) {
        ESP_LOGI(TAG, "✓ AI model loaded (%u bytes, %.1f KB)",
                 gru_model_data_len,
                 (float)gru_model_data_len / 1024.0f);
    } else {
        ESP_LOGW(TAG, "⚠ AI model empty — running in sensor-only mode");
    }

    ESP_LOGI(TAG, "\nCreating FreeRTOS tasks...");

    xTaskCreatePinnedToCore(
        sensor_task,                   
        "sensor_task",                 
        SENSOR_TASK_STACK_SIZE,        
        NULL,                          
        SENSOR_TASK_PRIORITY,          
        &sensor_task_handle,           
        CORE_0                         
    );
    ESP_LOGI(TAG, "  ✓ sensor_task (Core 0, priority %d)", SENSOR_TASK_PRIORITY);

    xTaskCreatePinnedToCore(
        motion_gate_task,
        "motion_task",
        MOTION_TASK_STACK_SIZE,
        NULL,
        MOTION_TASK_PRIORITY,
        &motion_task_handle,
        CORE_0
    );
    ESP_LOGI(TAG, "  ✓ motion_task (Core 0, priority %d)", MOTION_TASK_PRIORITY);

    ESP_LOGI(TAG, "\n╔══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   Smart Clip firmware running         ║");
    ESP_LOGI(TAG, "║   Sampling every 10 minutes           ║");
    ESP_LOGI(TAG, "║   BLE advertising as SmartClip        ║");
    ESP_LOGI(TAG, "╚══════════════════════════════════════╝\n");
}
