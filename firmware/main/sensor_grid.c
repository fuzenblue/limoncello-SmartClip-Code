#include "sensor_grid.h"
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "driver/i2c.h"
#include "driver/i2s_std.h"
#include "esp_timer.h"
#include "esp_adc/adc_oneshot.h"

static const char *TAG = "SENSOR_GRID";


#define BME680_REG_CHIP_ID      0xD0  
#define BME680_REG_CTRL_MEAS    0x74  
#define BME680_REG_CTRL_HUM     0x72  
#define BME680_REG_CONFIG       0x75  
#define BME680_REG_CTRL_GAS0    0x70  
#define BME680_REG_CTRL_GAS1    0x71  
#define BME680_REG_GAS_WAIT0    0x64  
#define BME680_REG_GAS_RES_H0   0x6C  
#define BME680_REG_STATUS       0x1D  
#define BME680_REG_PRESS_MSB    0x1F  
#define BME680_REG_TEMP_MSB     0x22  
#define BME680_REG_HUM_MSB      0x25  
#define BME680_REG_GAS_MSB      0x2A  


#define BME680_OS_NONE          0x00
#define BME680_OS_1X            0x01
#define BME680_OS_2X            0x02
#define BME680_OS_4X            0x03
#define BME680_OS_8X            0x04
#define BME680_OS_16X           0x05


#define BME680_MODE_SLEEP       0x00
#define BME680_MODE_FORCED      0x01


#define BME680_FILTER_OFF       0x00
#define BME680_FILTER_3         0x02  


#define BME680_CHIP_ID          0x61




static esp_err_t bme680_write_reg(uint8_t reg, uint8_t data)
{
    uint8_t buf[2] = { reg, data };
    return i2c_master_write_to_device(I2C_MASTER_NUM, BME680_I2C_ADDR,
                                       buf, 2, pdMS_TO_TICKS(100));
}


static esp_err_t bme680_read_reg(uint8_t reg, uint8_t *data, size_t len)
{
    return i2c_master_write_read_device(I2C_MASTER_NUM, BME680_I2C_ADDR,
                                         &reg, 1, data, len,
                                         pdMS_TO_TICKS(100));
}



static struct {
    
    uint16_t par_t1;
    int16_t  par_t2;
    int8_t   par_t3;

    
    uint16_t par_p1;
    int16_t  par_p2, par_p4, par_p5, par_p8, par_p9;
    int8_t   par_p3, par_p6, par_p7, par_p10;

    
    uint16_t par_h1, par_h2;
    int8_t   par_h3, par_h4, par_h5;
    uint8_t  par_h6, par_h7;

    
    int8_t   par_gh1, par_gh3;
    int16_t  par_gh2;

    
    float    t_fine;
} bme680_cal;


static bool bme680_cal_loaded = false;




esp_err_t bme680_init(void)
{
    esp_err_t ret;

    
    uint8_t chip_id;
    ret = bme680_read_reg(BME680_REG_CHIP_ID, &chip_id, 1);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "BME680: I2C read failed — check wiring!");
        return ret;
    }
    if (chip_id != BME680_CHIP_ID) {
        ESP_LOGE(TAG, "BME680: Unexpected chip ID 0x%02X (expected 0x%02X)",
                 chip_id, BME680_CHIP_ID);
        return ESP_ERR_INVALID_RESPONSE;
    }
    ESP_LOGI(TAG, "BME680: Chip ID verified (0x%02X)", chip_id);

    
    uint8_t cal_data[42];
    ret = bme680_read_reg(0x89, cal_data, 25);  
    if (ret != ESP_OK) return ret;

    uint8_t cal_data2[16];
    ret = bme680_read_reg(0xE1, cal_data2, 16);  
    if (ret != ESP_OK) return ret;

    
    bme680_cal.par_t1 = (uint16_t)(cal_data[0] | (cal_data[1] << 8));
    bme680_cal.par_t2 = (int16_t)(cal_data[2] | (cal_data[3] << 8));
    bme680_cal.par_t3 = (int8_t)cal_data[4];

    bme680_cal.par_p1 = (uint16_t)(cal_data[5] | (cal_data[6] << 8));
    bme680_cal.par_p2 = (int16_t)(cal_data[7] | (cal_data[8] << 8));
    bme680_cal.par_p3 = (int8_t)cal_data[9];

    bme680_cal_loaded = true;

    

    
    ret = bme680_write_reg(BME680_REG_CTRL_HUM, BME680_OS_1X);
    if (ret != ESP_OK) return ret;

    
    ret = bme680_write_reg(BME680_REG_CONFIG, BME680_FILTER_3 << 2);
    if (ret != ESP_OK) return ret;

    
    ret = bme680_write_reg(BME680_REG_GAS_RES_H0, 0xFF);  
    if (ret != ESP_OK) return ret;
    ret = bme680_write_reg(BME680_REG_GAS_WAIT0, 0x59);   
    if (ret != ESP_OK) return ret;

    
    ret = bme680_write_reg(BME680_REG_CTRL_GAS1, 0x10);
    if (ret != ESP_OK) return ret;

    ESP_LOGI(TAG, "BME680: Configuration complete (T×2, P×4, H×1, IIR=3, Gas=320°C/150ms)");
    return ESP_OK;
}


esp_err_t bme680_read(bme680_data_t *data)
{
    esp_err_t ret;

    if (!bme680_cal_loaded) {
        ESP_LOGE(TAG, "BME680: Calibration not loaded — call bme680_init() first");
        return ESP_ERR_INVALID_STATE;
    }

    
    uint8_t ctrl_meas = (BME680_OS_2X << 5) | (BME680_OS_4X << 2) | BME680_MODE_FORCED;
    ret = bme680_write_reg(BME680_REG_CTRL_MEAS, ctrl_meas);
    if (ret != ESP_OK) return ret;

    
    vTaskDelay(pdMS_TO_TICKS(250));

    
    uint8_t status;
    ret = bme680_read_reg(BME680_REG_STATUS, &status, 1);
    if (ret != ESP_OK) return ret;
    if (!(status & 0x80)) {
        ESP_LOGW(TAG, "BME680: Measurement not ready (status=0x%02X)", status);
        
    }

    
    uint8_t raw[8];
    ret = bme680_read_reg(BME680_REG_PRESS_MSB, raw, 8);
    if (ret != ESP_OK) return ret;

    uint32_t adc_press = ((uint32_t)raw[0] << 12) | ((uint32_t)raw[1] << 4) | (raw[2] >> 4);
    uint32_t adc_temp  = ((uint32_t)raw[3] << 12) | ((uint32_t)raw[4] << 4) | (raw[5] >> 4);
    uint16_t adc_hum   = ((uint16_t)raw[6] << 8) | raw[7];

    

    
    float var1 = ((float)adc_temp / 16384.0f - (float)bme680_cal.par_t1 / 1024.0f) *
                 (float)bme680_cal.par_t2;
    float var2 = (((float)adc_temp / 131072.0f - (float)bme680_cal.par_t1 / 8192.0f) *
                  ((float)adc_temp / 131072.0f - (float)bme680_cal.par_t1 / 8192.0f)) *
                 ((float)bme680_cal.par_t3 * 16.0f);
    bme680_cal.t_fine = var1 + var2;
    data->temp_celsius = bme680_cal.t_fine / 5120.0f;

    
    var1 = (bme680_cal.t_fine / 2.0f) - 64000.0f;
    var2 = var1 * var1 * ((float)bme680_cal.par_p6) / 131072.0f;
    var2 = var2 + var1 * ((float)bme680_cal.par_p5) * 2.0f;
    var2 = var2 / 4.0f + ((float)bme680_cal.par_p4) * 65536.0f;
    var1 = (((float)bme680_cal.par_p3 * var1 * var1) / 16384.0f +
            ((float)bme680_cal.par_p2 * var1)) / 524288.0f;
    var1 = (1.0f + var1 / 32768.0f) * (float)bme680_cal.par_p1;
    if (var1 > 0.0f) {
        data->pressure_hpa = 1048576.0f - (float)adc_press;
        data->pressure_hpa = ((data->pressure_hpa - var2 / 4096.0f) * 6250.0f) / var1;
        data->pressure_hpa /= 100.0f;   
    } else {
        data->pressure_hpa = 1013.25f;   
    }

    
    float temp_comp = bme680_cal.t_fine / 5120.0f;
    float var_h = (float)adc_hum - ((float)bme680_cal.par_h1 * 16.0f);
    var_h -= ((float)bme680_cal.par_h3 / 2.0f) * temp_comp;
    data->humidity_pct = var_h * 100.0f / 65536.0f;
    if (data->humidity_pct > 100.0f) data->humidity_pct = 100.0f;
    if (data->humidity_pct < 0.0f)   data->humidity_pct = 0.0f;

    
    uint8_t gas_raw[2];
    ret = bme680_read_reg(BME680_REG_GAS_MSB, gas_raw, 2);
    if (ret == ESP_OK) {
        uint16_t gas_adc = ((uint16_t)(gas_raw[0]) << 2) | (gas_raw[1] >> 6);
        uint8_t gas_range = gas_raw[1] & 0x0F;
        
        float range_factor = 1.0f;
        if (gas_range < 16) {
            
            static const float range_table[] = {
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.99f, 1.0f, 0.992f,
                1.0f, 1.0f, 0.998f, 0.995f, 1.0f, 0.99f, 1.0f, 1.0f
            };
            range_factor = range_table[gas_range];
        }
        data->gas_resistance_ohm = (float)gas_adc * range_factor * 1000.0f;
        data->gas_valid = true;
    } else {
        data->gas_resistance_ohm = 50000.0f;  
        data->gas_valid = false;
    }

    return ESP_OK;
}





static i2s_chan_handle_t rx_chan = NULL;

esp_err_t inmp441_init(void)
{

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(
        I2S_NUM_0,           
        I2S_ROLE_MASTER      
    );
    esp_err_t ret = i2s_new_channel(&chan_cfg, NULL, &rx_chan);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "INMP441: Failed to create I2S channel");
        return ret;
    }

    
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(INMP441_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT,   
            I2S_SLOT_MODE_MONO          
        ),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,     
            .bclk = INMP441_SCK_PIN,     
            .ws   = INMP441_WS_PIN,      
            .dout = I2S_GPIO_UNUSED,     
            .din  = INMP441_SD_PIN,      
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };

    ret = i2s_channel_init_std_mode(rx_chan, &std_cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "INMP441: Failed to configure I2S mode");
        return ret;
    }

    ret = i2s_channel_enable(rx_chan);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "INMP441: Failed to enable I2S channel");
        return ret;
    }

    ESP_LOGI(TAG, "INMP441: Initialised (SR=%dHz, 32-bit, mono)",
             INMP441_SAMPLE_RATE);
    return ESP_OK;
}


size_t inmp441_read_frame(int32_t *buffer, size_t samples)
{
    
    size_t bytes_to_read = samples * sizeof(int32_t);
    size_t bytes_read = 0;

    esp_err_t ret = i2s_channel_read(rx_chan, buffer, bytes_to_read,
                                      &bytes_read, pdMS_TO_TICKS(1000));
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "INMP441: Read error (%s)", esp_err_to_name(ret));
        return 0;
    }

    size_t samples_read = bytes_read / sizeof(int32_t);

    for (size_t i = 0; i < samples_read; i++) {
        buffer[i] >>= 8;   
    }

    return samples_read;
}

static uint16_t adc_buffer[PHOTODIODE_BUF_SIZE];
static volatile uint32_t adc_head = 0;  


static adc_oneshot_unit_handle_t adc_handle = NULL;


static esp_timer_handle_t adc_timer = NULL;

static void IRAM_ATTR photodiode_timer_isr(void *arg)
{
    
    int raw = 0;
    adc_oneshot_read(adc_handle, PHOTODIODE_ADC_CHANNEL, &raw);

    
    adc_buffer[adc_head] = (uint16_t)raw;
    adc_head = (adc_head + 1) % PHOTODIODE_BUF_SIZE;
}


esp_err_t photodiode_init(void)
{
    
    adc_oneshot_unit_init_cfg_t init_cfg = {
        .unit_id = ADC_UNIT_1,               
    };
    esp_err_t ret = adc_oneshot_new_unit(&init_cfg, &adc_handle);
    if (ret != ESP_OK) return ret;

    adc_oneshot_chan_cfg_t chan_cfg = {
        .atten = ADC_ATTEN_DB_11,            
        .bitwidth = ADC_BITWIDTH_12,         
    };
    ret = adc_oneshot_config_channel(adc_handle, PHOTODIODE_ADC_CHANNEL, &chan_cfg);
    if (ret != ESP_OK) return ret;

    esp_timer_create_args_t timer_args = {
        .callback = photodiode_timer_isr,
        .name = "adc_800hz",
        .dispatch_method = ESP_TIMER_TASK, 
    };
    ret = esp_timer_create(&timer_args, &adc_timer);
    if (ret != ESP_OK) return ret;

    
    ret = esp_timer_start_periodic(adc_timer, 1250);
    if (ret != ESP_OK) return ret;

    
    memset(adc_buffer, 0, sizeof(adc_buffer));
    adc_head = 0;

    ESP_LOGI(TAG, "Photodiode: ADC initialised (Ch0, 12-bit, 800Hz timer)");
    return ESP_OK;
}


void photodiode_get_buffer(uint16_t **buffer, volatile uint32_t *head)
{
    *buffer = adc_buffer;
    *head = adc_head;
}


#define MPU6050_REG_PWR_MGMT_1    0x6B  
#define MPU6050_REG_ACCEL_CONFIG  0x1C  
#define MPU6050_REG_MOT_THR      0x1F  
#define MPU6050_REG_MOT_DUR      0x20  
#define MPU6050_REG_ACCEL_XOUT_H 0x3B  
#define MPU6050_REG_WHO_AM_I     0x75  


#define MPU6050_ACCEL_SCALE      (2.0f / 32768.0f)


static esp_err_t mpu6050_write_reg(uint8_t reg, uint8_t data)
{
    uint8_t buf[2] = { reg, data };
    return i2c_master_write_to_device(I2C_MASTER_NUM, MPU6050_I2C_ADDR,
                                       buf, 2, pdMS_TO_TICKS(100));
}


static esp_err_t mpu6050_read_reg(uint8_t reg, uint8_t *data, size_t len)
{
    return i2c_master_write_read_device(I2C_MASTER_NUM, MPU6050_I2C_ADDR,
                                         &reg, 1, data, len,
                                         pdMS_TO_TICKS(100));
}


esp_err_t imu_init(void)
{
    esp_err_t ret;


    ret = mpu6050_write_reg(MPU6050_REG_PWR_MGMT_1, 0x00);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "MPU-6050: Wake failed — check wiring!");
        return ret;
    }
    vTaskDelay(pdMS_TO_TICKS(100));  

    
    uint8_t who_am_i;
    ret = mpu6050_read_reg(MPU6050_REG_WHO_AM_I, &who_am_i, 1);
    if (ret != ESP_OK) return ret;
    ESP_LOGI(TAG, "MPU-6050: WHO_AM_I = 0x%02X (expected 0x68)", who_am_i);

    ret = mpu6050_write_reg(MPU6050_REG_ACCEL_CONFIG, 0x00);  
    if (ret != ESP_OK) return ret;

    ret = mpu6050_write_reg(MPU6050_REG_MOT_THR, 25);  
    if (ret != ESP_OK) return ret;

    ret = mpu6050_write_reg(MPU6050_REG_MOT_DUR, 10);  
    if (ret != ESP_OK) return ret;

    ESP_LOGI(TAG, "MPU-6050: Initialised (±2g, motion threshold=50mg)");
    return ESP_OK;
}


esp_err_t imu_read_accel(imu_accel_t *data)
{
    uint8_t raw[6];
    esp_err_t ret = mpu6050_read_reg(MPU6050_REG_ACCEL_XOUT_H, raw, 6);
    if (ret != ESP_OK) return ret;

    int16_t raw_x = (int16_t)((raw[0] << 8) | raw[1]);
    int16_t raw_y = (int16_t)((raw[2] << 8) | raw[3]);
    int16_t raw_z = (int16_t)((raw[4] << 8) | raw[5]);

    data->ax = (float)raw_x * MPU6050_ACCEL_SCALE;
    data->ay = (float)raw_y * MPU6050_ACCEL_SCALE;
    data->az = (float)raw_z * MPU6050_ACCEL_SCALE;


    data->mag = sqrtf(data->ax * data->ax +
                       data->ay * data->ay +
                       data->az * data->az);

    return ESP_OK;
}
