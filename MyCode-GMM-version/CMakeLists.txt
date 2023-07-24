# cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
# mingw32-make
# .\main.exe ../../../data/sheep.jpg
cmake_minimum_required(VERSION 3.10)
project(MyCode-GMM-version)
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
include_directories(core BorderMatting GrabCut GMM maxflow)


# add graph library
add_library(graph SHARED core/GrabCut/maxflow/graph.cpp)
# add maxflow library --> graph library
add_library(maxflow SHARED core/GrabCut/maxflow/maxflow.cpp)
target_link_libraries(maxflow graph)

# add GMM library
add_library(GMM SHARED core/GrabCut/GMM/GMM.cpp)
target_link_libraries(GMM ${OpenCV_LIBS})
# add GrabCut library --> GMM, graph, maxflow library
add_library(GrabCut SHARED core/GrabCut/GrabCut.cpp)
target_link_libraries(GrabCut GMM graph maxflow)

# add BorderMatting library
add_library(BorderMatting SHARED core/BorderMatting/BorderMatting.cpp)
target_link_libraries(BorderMatting ${OpenCV_LIBS})

# add MyCode-GCApplication library --> GrabCut, BorderMatting library
add_library(MyCode-GCApplication SHARED core/MyCode-GCApplication.cpp)
target_link_libraries(MyCode-GCApplication ${OpenCV_LIBS} GrabCut BorderMatting)
# add main library --> MyCode-GCApplication library
add_executable(main MyCode-main.cpp)
target_link_libraries(main ${OpenCV_LIBS} MyCode-GCApplication)
