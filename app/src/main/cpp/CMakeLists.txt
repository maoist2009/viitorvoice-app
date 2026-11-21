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
    mstore.cpp
    vv_pipeline.cpp
    vv_nar.cpp
    vv_encode.cpp
    vv_mel.cpp
)

find_library(log-lib log)

# libasr_qnn.so — SenseVoice-Small / Silero VAD / Qwen3-ForcedAligner.
# The sources are shared with ~/projects/asr (same QNN executor, separate copy
# so the two .so files do not share global state).
add_library(asr_qnn SHARED
    asr/asr_jni.c
    asr/asr_engine.c
    asr/aligner_jni.c
    asr/aligner_engine.c
    asr/asr_mel.c
    asr/sv_engine.c
    asr/sv_frontend.c
    asr/cmp_engine.c
    asr/cmp_frontend.c
    asr/vad_engine.c
    asr/vad_jni.c
    asr/qnn_executor.cpp
)
target_include_directories(asr_qnn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/asr)
target_link_libraries(asr_qnn PRIVATE ${log-lib} dl android)

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
