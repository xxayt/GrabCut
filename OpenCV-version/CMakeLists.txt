# cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
# mingw32-make
# .\GrabCut.exe ../../../data/sheep.jpg
cmake_minimum_required(VERSION 3.10)
project(OpenCV-grabcut)
set(CMAKE_CXX_STANDARD 11)

# ./build: ${PROJECT_BINARY_DIR}
# ./: ${CMAKE_SOURCE_DIR}
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin/debug)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/release)

# Debug mode
if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0")
    message("Debug mode:${CMAKE_C_FLAGS_DEBUG}")
# Release mode
elseif(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -O3")
    message("Release mode:${CMAKE_C_FLAGS_RELEASE}")
# Default mode
else()
    message("else:${CMAKE_BUILD_TYPE}")
    message("else:${CMAKE_C_FLAGS_RELEASE}")
endif()



# include the directory itself as a path to include directories
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(core)

add_library(OpenCV-GCApplication SHARED core/OpenCV-GCApplication.cpp)
target_link_libraries(OpenCV-GCApplication ${OpenCV_LIBS})

add_executable(GrabCut OpenCV-main.cpp)
target_link_libraries(GrabCut ${OpenCV_LIBS} OpenCV-GCApplication)
