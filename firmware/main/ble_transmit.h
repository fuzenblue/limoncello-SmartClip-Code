
#ifndef BLE_TRANSMIT_H
#define BLE_TRANSMIT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>




#define BLE_DEVICE_NAME         "SmartClip"
#define BLE_SERVICE_UUID        0xFE40
#define BLE_CHAR_FV_UUID        0xFE41   
#define BLE_CHAR_OTA_UUID       0xFE42   


#define BLE_MAX_MTU             512


void ble_init(void);


bool ble_is_connected(void);


int ble_transmit_feature_vector(const uint8_t *data, size_t len);

#endif 
