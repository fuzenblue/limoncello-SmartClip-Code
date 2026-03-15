#ifndef SENSOR_GRID_H
#define SENSOR_GRID_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"


#define I2C_MASTER_NUM          I2C_NUM_0    
#define I2C_SDA_PIN             GPIO_NUM_21  
#define I2C_SCL_PIN             GPIO_NUM_22  
#define I2C_MASTER_FREQ_HZ     400000       


#define BME680_I2C_ADDR         0x76         
                                              


#define MPU6050_I2C_ADDR        0x68         


#define INMP441_WS_PIN          GPIO_NUM_25  
#define INMP441_SCK_PIN         GPIO_NUM_26  
#define INMP441_SD_PIN          GPIO_NUM_34  
#define INMP441_SAMPLE_RATE     16000        


#define PHOTODIODE_ADC_CHANNEL  ADC_CHANNEL_0  
#define PHOTODIODE_SAMPLE_RATE  800          
#define PHOTODIODE_BUF_SIZE     512          


typedef struct {
    float pressure_hpa;          
    float temp_celsius;          
    float humidity_pct;          
    float gas_resistance_ohm;    
    bool  gas_valid;             
} bme680_data_t;


typedef struct {
    float ax;    
    float ay;    
    float az;    
    float mag;   
} imu_accel_t;

esp_err_t bme680_init(void);
esp_err_t bme680_read(bme680_data_t *data);
esp_err_t inmp441_init(void);

size_t inmp441_read_frame(int32_t *buffer, size_t samples);

esp_err_t photodiode_init(void);
void photodiode_get_buffer(uint16_t **buffer, volatile uint32_t *head);

esp_err_t imu_init(void);
esp_err_t imu_read_accel(imu_accel_t *data);

#endif
