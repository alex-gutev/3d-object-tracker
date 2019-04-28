cmake_minimum_required(VERSION 3.7)
project(3dtracker)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

option(2D_TRACKER "Compile with OpenCV's 2D mean-shift tracker" OFF)
option(PARALLEL "Compile parallelised 3D tracker" ON)

if(TRACKER_2D)
    add_definitions(-DFYP_TRACKER_2D)
endif()

if(PARALLEL)
    add_definitions(-DFYP_TRACKER_PARALLEL)
endif()

add_executable(tracker main.cpp yuv_reader.cpp yuv_reader.h tracker_app.cpp tracker_app.h tracker.cpp tracker.h view.cpp view.h view_parser.cpp view_parser.h)
target_link_libraries(tracker ${OpenCV_LIBS})

