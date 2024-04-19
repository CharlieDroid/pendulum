//
// Created by Charles on 2/5/2024.
//

#ifndef RPI_CPP_GLOBALS_H
#define RPI_CPP_GLOBALS_H

//#define PRECHECK
// Turns off motor during precheck
//#define PRECHECK_NO_MOTOR
//#define DEBUG
#include <iostream>

#ifndef DEBUG
#include <boost/asio.hpp>

using namespace boost::asio;
// global variables e.g. used by environment.cpp and main.cpp
extern io_context io;
extern executor_work_guard<io_context::executor_type> work;
extern steady_timer timer;
constexpr float DT{ 0.02f };  // time delta
constexpr int DT_MS{ static_cast<int>(DT * 1000.0f) };
constexpr int TOTAL_TIMESTEPS{ 1000 };
#ifndef PRECHECK
constexpr int EPISODES{ 1000 };
#else
constexpr int EPISODES{ 1 };
#endif  // PRECHECK
#endif  // DEBUG

extern std::string ACTOR_FILENAME;
extern std::string MEMORY_FILENAME;
extern std::string ACTOR_FILEPATH;

#endif //RPI_CPP_GLOBALS_H
