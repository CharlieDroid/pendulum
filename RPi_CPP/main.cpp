//
// Created by Charles on 2/3/2024.
//
#include "globals.h"
#include "environment.h"
#include "agent.h"

#include <Eigen/Dense>
#include <pigpio.h>
#include <boost/asio.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>


using namespace boost::asio;
static Eigen::MatrixXf observations(TOTAL_TIMESTEPS + 1, INPUT_SIZE);
static Eigen::MatrixXf actions(TOTAL_TIMESTEPS, OUTPUT_SIZE);
static Eigen::VectorXf observation(INPUT_SIZE);
static Agent actor{ initActor() };  // not const because I update this every time
static int timestep{ 0 };
io_context io;
executor_work_guard<io_context::executor_type> work{ make_work_guard(io) };
steady_timer timer(io, std::chrono::milliseconds(DT_MS));

#ifdef PRECHECK  // edit precheck function to have similar structure to the one below
void doEpisode(const system::error_code& ec)
{
    // moving pendulum moves the motor
    printEncoderValues();

    timestep++;
    if (timestep < TOTAL_TIMESTEPS)
    {
        // Create a new timer with the same interval (20ms)
        asio::deadline_timer *timer;
        timer = new asio::deadline_timer(io, posix_time::milliseconds(DT_MS));
        timer->async_wait(&doEpisode);
    } else  // stop loop
    { io.stop(); io.restart(); }
}
#else
void doEpisode([[maybe_unused]] const boost::system::error_code&)
{
    observation = getObservation();
    // takes about 1.2e-5 seconds to feedForward
    float mu{ feedForward(actor, observation) };
    step(mu);
    // save data
    observations.row(timestep++) = observation;
    actions(timestep - 2, 0) = mu;

    if (timestep < (TOTAL_TIMESTEPS + 1))
    {
        // Create a new timer with the same interval (20ms)
        timer.expires_after(std::chrono::milliseconds(DT_MS));
        timer.async_wait(&doEpisode);
    } else  // stop loop
    { timestep = 0; rotate(0); io.stop(); }
}
#endif

int main()
{
    try
    {
        // ==========[ Initialize ]==========
        // sets the clock to 1us
        // gpioCfgClock(sample rate (us), 0 (PWM) or 1 (PCM), uint deprecated)
        gpioCfgClock(1, 0, 0);
        if (gpioInitialise() < 0) return 1;
        initEnv();
        {  // delete actor zip file if it exists before program starts, it should only be csv files
            std::ifstream file(ACTOR_FILEPATH);
            if (file.good())
            {
                if (std::system(("cd models; rm -f " + ACTOR_FILENAME).c_str()) != 0)
                { std::cout << "Error deleting actor file.\n"; }
            }
        }

        // ==========[ Run Episode ]==========
        for (int ep{ 0 }; ep < EPISODES; ++ep)
        {
            std::cout << "episode " << ep << "\n";
            observation = reset();
            observations.row(timestep++) = observation;
            float mu{ feedForward(actor, observation) };
            step(mu);  // step first to move then get observation after one time step

            // Run episode with DT_MS sampling rate
            timer.expires_after(std::chrono::milliseconds(DT_MS));
            timer.async_wait(&doEpisode);
            io.run();

            work.reset();
            io.restart();
#ifndef PRECHECK
            // reset environment after episode
            // convert the data to csv
            saveMemory(observations, actions);

            std::cout << "...waiting for model...\n";
            rotate(0);  // make sure motor is not moving
            while (true)
            {
                std::ifstream file(ACTOR_FILEPATH);
                if (file.good())  // if actor file exists
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    std::cout << "Unzipping " << ACTOR_FILENAME.c_str() << "\n";
                    int returnCode{std::system(("cd models; unzip -o " + ACTOR_FILENAME).c_str())};  // unzip actor file
                    if (returnCode != 0) std::perror("Error unzipping actor file\n");
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    actor = initActor();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
#endif
        }

        // ==========[ Kill Everything ]==========
        killEnv();
        gpioTerminate();
    }
    catch(const std::exception& e)
    {
        // if there is error ensures motor stops moving, upon application motor still moves
        killEnv();
        gpioTerminate();
        std::cerr << e.what() << "\n";
    }

    return 0;
}
