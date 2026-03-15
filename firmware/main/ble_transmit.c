

#include "ble_transmit.h"
#include <string.h>
#include "esp_log.h"
#include "esp_bt.h"
#include "esp_bt_main.h"
#include "esp_gap_ble_api.h"
#include "esp_gatts_api.h"
#include "esp_bt_defs.h"

static const char *TAG = "BLE_TX";


static bool ble_connected = false;
static uint16_t ble_conn_id = 0;         
static uint16_t ble_gatts_if = 0;        
static uint16_t fv_char_handle = 0;      
static bool     notifications_enabled = false;




static uint8_t service_uuid[2] = {
    (BLE_SERVICE_UUID & 0xFF),
    (BLE_SERVICE_UUID >> 8) & 0xFF
};


static uint8_t char_fv_uuid[2] = {
    (BLE_CHAR_FV_UUID & 0xFF),
    (BLE_CHAR_FV_UUID >> 8) & 0xFF
};


static uint8_t fv_char_prop = ESP_GATT_CHAR_PROP_BIT_READ |
                               ESP_GATT_CHAR_PROP_BIT_NOTIFY;


static uint16_t fv_cccd = 0x0000;


static uint8_t fv_value[BLE_MAX_MTU];
static uint16_t fv_value_len = 0;





static esp_ble_adv_data_t adv_data = {
    .set_scan_rsp = false,
    .include_name = true,             
    .include_txpower = true,          
    .min_interval = 0x0006,           
    .max_interval = 0x0010,           
    .appearance = 0x00,               
    .manufacturer_len = 0,
    .p_manufacturer_data = NULL,
    .service_data_len = 0,
    .p_service_data = NULL,
    .service_uuid_len = sizeof(service_uuid),
    .p_service_uuid = service_uuid,
    .flag = (ESP_BLE_ADV_FLAG_GEN_DISC |     
             ESP_BLE_ADV_FLAG_BREDR_NOT_SPT), 
};


static esp_ble_adv_params_t adv_params = {
    .adv_int_min = 0x20,              
    .adv_int_max = 0x40,              
    .adv_type = ADV_TYPE_IND,         
    .own_addr_type = BLE_ADDR_TYPE_PUBLIC,
    .channel_map = ADV_CHNL_ALL,      
    .adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
};





static void gap_event_handler(esp_gap_ble_cb_event_t event,
                                esp_ble_gap_cb_param_t *param)
{
    switch (event) {
    case ESP_GAP_BLE_ADV_DATA_SET_COMPLETE_EVT:
        
        esp_ble_gap_start_advertising(&adv_params);
        ESP_LOGI(TAG, "BLE advertising started");
        break;

    case ESP_GAP_BLE_ADV_START_COMPLETE_EVT:
        if (param->adv_start_cmpl.status != ESP_BT_STATUS_SUCCESS) {
            ESP_LOGE(TAG, "Advertising start failed: %d",
                     param->adv_start_cmpl.status);
        }
        break;

    case ESP_GAP_BLE_UPDATE_CONN_PARAMS_EVT:
        ESP_LOGI(TAG, "Connection params updated: interval=%d latency=%d timeout=%d",
                 param->update_conn_params.conn_int,
                 param->update_conn_params.latency,
                 param->update_conn_params.timeout);
        break;

    default:
        break;
    }
}



static void gatts_event_handler(esp_gatts_cb_event_t event,
                                 esp_gatt_if_t gatts_if,
                                 esp_ble_gatts_cb_param_t *param)
{
    switch (event) {
    case ESP_GATTS_REG_EVT:
        
        ESP_LOGI(TAG, "GATT app registered (if=%d)", gatts_if);
        ble_gatts_if = gatts_if;

        esp_ble_gap_set_device_name(BLE_DEVICE_NAME);
        esp_ble_gap_config_adv_data(&adv_data);

        
        esp_gatt_srvc_id_t service_id = {
            .is_primary = true,
            .id = {
                .inst_id = 0,
                .uuid = {
                    .len = ESP_UUID_LEN_16,
                    .uuid = { .uuid16 = BLE_SERVICE_UUID },
                },
            },
        };
        esp_ble_gatts_create_service(gatts_if, &service_id, 8);
        break;

    case ESP_GATTS_CREATE_EVT:
        
        ESP_LOGI(TAG, "Service created (handle=%d)", param->create.service_handle);
        esp_ble_gatts_start_service(param->create.service_handle);

        
        esp_bt_uuid_t char_uuid = {
            .len = ESP_UUID_LEN_16,
            .uuid = { .uuid16 = BLE_CHAR_FV_UUID },
        };

        esp_attr_value_t char_val = {
            .attr_max_len = BLE_MAX_MTU,
            .attr_len = 0,
            .attr_value = fv_value,
        };

        esp_err_t ret = esp_ble_gatts_add_char(
            param->create.service_handle,
            &char_uuid,
            ESP_GATT_PERM_READ,          
            fv_char_prop,                 
            &char_val,
            NULL                          
        );
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to add characteristic: %s",
                     esp_err_to_name(ret));
        }
        break;

    case ESP_GATTS_ADD_CHAR_EVT:
        
        fv_char_handle = param->add_char.attr_handle;
        ESP_LOGI(TAG, "FV characteristic added (handle=%d)", fv_char_handle);
        break;

    case ESP_GATTS_CONNECT_EVT:
        
        ble_connected = true;
        ble_conn_id = param->connect.conn_id;
        ESP_LOGI(TAG, "Client connected (conn_id=%d, addr=%02x:%02x:%02x:%02x:%02x:%02x)",
                 ble_conn_id,
                 param->connect.remote_bda[0], param->connect.remote_bda[1],
                 param->connect.remote_bda[2], param->connect.remote_bda[3],
                 param->connect.remote_bda[4], param->connect.remote_bda[5]);

        
        esp_ble_gatt_set_local_mtu(BLE_MAX_MTU);
        break;

    case ESP_GATTS_DISCONNECT_EVT:
        
        ble_connected = false;
        notifications_enabled = false;
        ESP_LOGI(TAG, "Client disconnected (reason=0x%X). Restarting advertising.",
                 param->disconnect.reason);
        esp_ble_gap_start_advertising(&adv_params);
        break;

    case ESP_GATTS_WRITE_EVT:
        
        if (param->write.len == 2) {
            uint16_t cccd_val = param->write.value[0] | (param->write.value[1] << 8);
            if (cccd_val == 0x0001) {
                notifications_enabled = true;
                ESP_LOGI(TAG, "Notifications ENABLED by client");
            } else {
                notifications_enabled = false;
                ESP_LOGI(TAG, "Notifications DISABLED by client");
            }
        }

        
        if (param->write.need_rsp) {
            esp_ble_gatts_send_response(gatts_if, param->write.conn_id,
                                         param->write.trans_id,
                                         ESP_GATT_OK, NULL);
        }
        break;

    case ESP_GATTS_READ_EVT:
        
        {
            esp_gatt_rsp_t rsp;
            memset(&rsp, 0, sizeof(rsp));
            rsp.attr_value.handle = param->read.handle;
            rsp.attr_value.len = fv_value_len;
            if (fv_value_len > 0) {
                memcpy(rsp.attr_value.value, fv_value, fv_value_len);
            }
            esp_ble_gatts_send_response(gatts_if, param->read.conn_id,
                                         param->read.trans_id,
                                         ESP_GATT_OK, &rsp);
        }
        break;

    case ESP_GATTS_MTU_EVT:
        ESP_LOGI(TAG, "MTU negotiated: %d bytes", param->mtu.mtu);
        break;

    default:
        break;
    }
}




void ble_init(void)
{
    
    ESP_ERROR_CHECK(esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT));

    
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_bt_controller_init(&bt_cfg));
    ESP_ERROR_CHECK(esp_bt_controller_enable(ESP_BT_MODE_BLE));

    
    ESP_ERROR_CHECK(esp_bluedroid_init());
    ESP_ERROR_CHECK(esp_bluedroid_enable());

    
    ESP_ERROR_CHECK(esp_ble_gap_register_callback(gap_event_handler));
    ESP_ERROR_CHECK(esp_ble_gatts_register_callback(gatts_event_handler));

    
    ESP_ERROR_CHECK(esp_ble_gatts_app_register(0));

    ESP_LOGI(TAG, "BLE GATT server initialised (service=0x%04X, char=0x%04X)",
             BLE_SERVICE_UUID, BLE_CHAR_FV_UUID);
}


bool ble_is_connected(void)
{
    return ble_connected && notifications_enabled;
}


int ble_transmit_feature_vector(const uint8_t *data, size_t len)
{
    if (!ble_connected) {
        ESP_LOGW(TAG, "Cannot transmit — no BLE connection");
        return -1;
    }

    if (!notifications_enabled) {
        ESP_LOGW(TAG, "Cannot transmit — notifications not enabled by client");
        return -1;
    }

    if (len > BLE_MAX_MTU) {
        ESP_LOGE(TAG, "Payload too large (%zu > %d)", len, BLE_MAX_MTU);
        return -1;
    }

    
    memcpy(fv_value, data, len);
    fv_value_len = (uint16_t)len;

    
    esp_err_t ret = esp_ble_gatts_send_indicate(
        ble_gatts_if,       
        ble_conn_id,         
        fv_char_handle,      
        (uint16_t)len,       
        (uint8_t *)data,     
        false                
    );

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "NOTIFY failed: %s", esp_err_to_name(ret));
        return -1;
    }

    ESP_LOGI(TAG, "Transmitted %zu bytes via BLE NOTIFY", len);
    return 0;
}
