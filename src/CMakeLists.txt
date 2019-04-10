cmake_minimum_required (VERSION 2.6)
project (leptjson_test C)

add_library(h264lib h264-lab.c)
add_executable(encode_app minih264e_test.c)
target_link_libraries(encode_app h264lib)
