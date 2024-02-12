cmake_minimum_required(VERSION 3.27)
project(RPi_CPP)

set(CMAKE_CXX_STANDARD 17)

# prevent command prompt from popping up
#add_executable(RPi_CPP WIN32 main.cpp)
#add_executable(RPi_CPP main.cpp
#        agent.cpp
#        agent.h
#        environment.cpp
#        environment.h)
add_executable(RPi_CPP main.cpp
        agent.cpp
        agent.h
        environment.cpp
        globals.h)
# run standalone executable file
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/eigen-3.4.0)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/pigpio-master)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/boost_1_84_0)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/libzip-1.10.1/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
