cmake_minimum_required(VERSION 2.8)
project(open)
find_package( OpenCV REQUIRED )
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")

    message([main] " OpenMP_C_FLAGS=${OpenMP_C_FLAGS}") # -fopenmp
    message([main] " OpenMP_CXX_FLAGS}=${OpenMP_CXX_FLAGS}") # -fopenmp
    message([main] " OpenMP_EXE_LINKER_FLAGS=${OpenMP_EXE_LINKER_FLAGS}") # ***

    # no use for xxx_INCLUDE_DIRS and xxx_libraries for OpenMP
    message([main] " OpenMP_INCLUDE_DIRS=${OpenMP_INCLUDE_DIRS}") # ***
    message([main] " OpenMP_LIBRARIES=${OpenMP_LIBRARIES}") # ***

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
add_executable( open open.cpp )
target_link_libraries( open ${OpenCV_LIBS} )