//
// Created by Charles on 2/3/2024.
//
#include "globals.h"  // where you define preprocessor vars

#ifndef DEBUG
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
#else
#include "agent.h"

#include <Eigen/Dense>

#include <iostream>
#include <random>
#endif

#ifndef DEBUG
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
#endif  // PRECHECK
#endif  // DEBUG

int main()
{
#ifndef DEBUG
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
#else  // put sample code here
    // FOR SAC
    // 0.34740288703003497,-2.631653915502039,1.8083727152545235,2.520265384708056,         0.5868830680847168, -3.4594898223876953
    // 0.13585402704080962,0.006645742428664295,2.0695956258390398,-2.322754065655531       -0.07293418049812317, -3.0109152793884277
    Agent actor{ initActor() };
    // FOR TD3
    // -0.22245886494711836,2.883761735794429,-0.24658002892348066,1.11295463739689,        0.6580871939659119
    // -0.11689328349867878,-0.03068256840957561,1.094427414223052,-1.2609386850399005,     -0.26557686924934387
    Eigen::VectorXf observation(4);
    observation << -0.22212738909436847,0.5482226152932184,2.1219093606609887,-4.241789495499612;

//    VectorXf a{ (actor.fc1.weight * observation + actor.fc1.bias).array().max(0) };
//    a = (actor.fc2.weight * a + actor.fc2.bias).array().max(0);
//    double mu{ (actor.mu.weight * a + actor.mu.bias).value() };
//    std::cout << "mu: " << mu << "\n";
//    double sigma{ ((actor.sigma.weight * a + actor.sigma.bias).array().tanh()).value() };
//    sigma = exp(LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (sigma + 1));
//    std::cout << "sigma: " << sigma << "\n";
//
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::normal_distribution<double> distribution(mu, sigma);
//
//    float action{ static_cast<float>(tanh(distribution(gen))) };
    float action{ feedForward(actor, observation) };
    std::cout << "action: " << action << "\n";

//    std::cout << "output: " << feedForward(actor, observation) << "\n";
//    observation << -0.11689328349867878,-0.03068256840957561,1.094427414223052,-1.2609386850399005;
//    std::cout << "output: " << feedForward(actor, observation) << "\n";
#endif
    return 0;
}
