cmake_minimum_required(VERSION 3.22.1)
project(vv_qnn LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VV_LIB_NAME vv_qnn)

message(STATUS "QNN_SDK_DIR = ${QNN_SDK_DIR}")

include_directories(${QNN_SDK_DIR}/include/QNN)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ONNX Runtime (vendored from the onnxruntime-android AAR): headers here, the
# .so in app/src/main/jniLibs so the APK packaging picks it up.
set(ORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime)

add_compile_options(
    -O2
    -fexceptions
    -frtti
    -Wall
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-unused-function
)

# libvv_qnn.so — JNI bridge + QNN HTP executor + ORT encoder.
# The QNN .so are NOT linked: they are dlopen'd at runtime from the app's own
# lib dir, so the QNN SDK version can be swapped without relinking.
add_library(${VV_LIB_NAME} SHARED
    vv_jni.cpp
    qnn_executor.cpp
    vv_pipeline.cpp
    vv_nar.cpp
    vv_encode.cpp
    vv_mel.cpp
)

find_library(log-lib log)

# The log-mel frontend is the only hot pure-CPU code in the pipeline; let the
# compiler vectorize it (no -ffast-math: the mel values feed a calibrated graph).
set_source_files_properties(vv_mel.cpp PROPERTIES COMPILE_OPTIONS "-O3")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libonnxruntime.so)
target_include_directories(${VV_LIB_NAME} PRIVATE ${ORT_DIR}/include)

target_link_libraries(${VV_LIB_NAME} PRIVATE
    ${log-lib}
    dl
    android
    onnxruntime
)
