/**
 * ============================================================================
 * FILE B9: CMakeLists.txt (Project Root)
 * ============================================================================
 * PURPOSE: Top-level CMake file for the ESP-IDF project.
 *
 * ESP-IDF uses CMake as its build system.  This file tells CMake:
 *   1. Where to find the ESP-IDF framework
 *   2. What your project is called
 *   3. Which components (source folders) to build
 *
 * To build:
 *   idf.py set-target esp32s3
 *   idf.py menuconfig     # Configure SDK options (optional)
 *   idf.py build           # Compile the firmware
 *   idf.py flash            # Flash to the device
 *   idf.py monitor          # View serial output
 * ============================================================================
 */

# Minimum required CMake version (ESP-IDF v5.x requires 3.16+)
cmake_minimum_required(VERSION 3.16)

# Include the ESP-IDF project() function.
# The $ENV{IDF_PATH} environment variable points to the ESP-IDF installation.
# This is set automatically when you source the ESP-IDF export script:
#   . $IDF_PATH/export.sh       (Linux/macOS)
#   %IDF_PATH%\export.bat       (Windows)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# Project name — this becomes the firmware binary name:
#   build/smartclip_firmware.bin
project(smartclip_firmware)
