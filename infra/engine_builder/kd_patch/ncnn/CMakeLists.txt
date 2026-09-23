
if(MSVC)
    # warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    add_definitions(/wd4996)
endif()

add_executable(benchncnn benchncnn.cpp)
add_executable(benchncnn_kd_fp32 benchncnn_wo_warmup_fp32.cpp)
add_executable(benchncnn_kd_fp16 benchncnn_wo_warmup_fp16.cpp)
add_executable(benchncnn_kd_bf16 benchncnn_wo_warmup_bf16.cpp)
add_executable(benchncnn_kd_int8 benchncnn_wo_warmup_int8.cpp)
target_link_libraries(benchncnn PRIVATE ncnn)
target_link_libraries(benchncnn_kd_fp32 PRIVATE ncnn)
target_link_libraries(benchncnn_kd_fp16 PRIVATE ncnn)
target_link_libraries(benchncnn_kd_bf16 PRIVATE ncnn)
target_link_libraries(benchncnn_kd_int8 PRIVATE ncnn)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_libraries(benchncnn PRIVATE nodefs.js)
    target_link_libraries(benchncnn_kd_fp32 PRIVATE nodefs.js)
    target_link_libraries(benchncnn_kd_fp16 PRIVATE nodefs.js)
    target_link_libraries(benchncnn_kd_bf16 PRIVATE nodefs.js)
    target_link_libraries(benchncnn_kd_int8 PRIVATE nodefs.js)
endif()

# add benchncnn to a virtual project group
set_property(TARGET benchncnn PROPERTY FOLDER "benchmark")
set_property(TARGET benchncnn_kd_fp32 PROPERTY FOLDER "benchmark")
set_property(TARGET benchncnn_kd_fp16 PROPERTY FOLDER "benchmark")
set_property(TARGET benchncnn_kd_bf16 PROPERTY FOLDER "benchmark")
set_property(TARGET benchncnn_kd_int8 PROPERTY FOLDER "benchmark")
