cmake_minimum_required(VERSION 3.16)
project(rk3562_edge_vision LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Link-only stubs; the board supplies the real librknnrt.so / libturbojpeg.so.0 at runtime.
add_library(rknnrt SHARED rknn_stub.c)
add_library(turbojpeg SHARED tj_stub.c)
set_target_properties(turbojpeg PROPERTIES VERSION 0 SOVERSION 0 OUTPUT_NAME turbojpeg)

add_executable(rknn_cam rknn_cam.cc postprocess.cc)
target_compile_options(rknn_cam PRIVATE -O2 -pthread)
target_link_libraries(rknn_cam PRIVATE pthread rknnrt turbojpeg)

add_executable(rknn_yolov5_demo main.cc postprocess.cc)
target_compile_options(rknn_yolov5_demo PRIVATE -O2)
target_link_libraries(rknn_yolov5_demo PRIVATE rknnrt)

add_executable(v4l2_probe v4l2_probe.c)
add_executable(v4l2_capture v4l2_capture.c)
target_compile_options(v4l2_capture PRIVATE -O2 -pthread)
target_link_libraries(v4l2_capture PRIVATE pthread)