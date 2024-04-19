//
// Created by Charles on 2/3/2024.
//
#include "globals.h"
#include "agent.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <thread>
#ifdef SAC
#include <random>
#endif

using namespace Eigen;
std::string ACTOR_FILENAME{ "td3_fork_actor.zip" };
std::string MEMORY_FILENAME{ "td3_fork_memory.zip" };
std::string ACTOR_FILEPATH{ "models/" + ACTOR_FILENAME };

Layer initLayer(const std::string& filename, const int& outputSize, const int& inputSize)
{
    using namespace std;
    // init layer
    Layer layer{ MatrixXf(outputSize, inputSize), VectorXf(outputSize) };

    // init biases of layer, placed in code block to not interfere with weights init
    {
        string fullFilename{ filename + "_biases.csv" };
        ifstream file(static_cast<string>(fullFilename), ios::in);
        // wait until file exists
        while (!file.good())
        {
            this_thread::sleep_for(std::chrono::seconds(1));
            file = ifstream(static_cast<string>(fullFilename), ios::in);
        }
        if (!file.is_open()) cout << "Error opening " << fullFilename.c_str() << " file.\n";
        else
        {
            string line;
            for (int i{ 0 }; i < outputSize; ++i)  // output size is same size of bias
            {
                getline(file, line);
                stringstream ss(line);
                float neuron{ std::stof(line) }; // the converted float
                layer.bias(i) = neuron;
            }
            file.close();
#ifndef DEBUG
            // delete file after loading
            if (std::system(("rm -f " + fullFilename).c_str()) != 0)
            { cout << "Error deleting " << fullFilename.c_str() << " file.\n"; }
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // wait until deleted
#endif
        }
    }

    // init weights of layer
    string fullFilename{ filename + "_weights.csv"};
    ifstream file(static_cast<string>(fullFilename), ios::in);
    // wait until file exists
    while (!file.good())
    {
        this_thread::sleep_for(std::chrono::seconds(1));
        file = ifstream(static_cast<string>(fullFilename), ios::in);
    }
    if (!file.is_open()) cout << "Error opening " << fullFilename.c_str() << " file.\n";
    else
    {
        string line;
        for (int i{ 0 }; i < outputSize; ++i)
        {
            getline(file, line);  // get line by line split by \n
            stringstream ss(line);
            string token{};
            for (int j{ 0 }; j < inputSize; ++j)  // row size is input size for the weights
            {
                getline(ss, token, ',');
                float neuron{std::stof(token)}; // the converted float
                layer.weight(i, j) = neuron;
            }
        }
        file.close();
#ifndef DEBUG
        // delete file after loading
        if (std::system(("rm -f " + fullFilename).c_str()) != 0)
        { cout << "Error deleting " << fullFilename.c_str() << " file.\n"; }
#endif
    }
    return layer;
}

Agent initActor()
{
    Agent actor{};
#ifdef LAYER_NORMALIZATION
    std::cout << "...loading LN model...\n";
    actor.ln1 = initLayer("models/ln1", LAYER1_SIZE, 1);
    actor.ln2 = initLayer("models/ln2", LAYER2_SIZE, 1);
#else
    std::cout << "...loading model...\n";
#endif
    actor.fc1 = initLayer("models/fc1", LAYER1_SIZE, INPUT_SIZE);
    actor.fc2 = initLayer("models/fc2", LAYER2_SIZE, LAYER1_SIZE);
    actor.mu = initLayer("models/mu", OUTPUT_SIZE, LAYER2_SIZE);
#ifdef SAC
    actor.sigma = initLayer("models/sigma", OUTPUT_SIZE, LAYER2_SIZE);
#endif
#ifndef DEBUG
    if (std::system(("cd models; rm -f " + ACTOR_FILENAME).c_str()) != 0)
    { std::cout << "Error deleting actor file.\n"; }
#endif
    return actor;
}

#ifdef SAC
float feedForward(const Agent& actor, const VectorXf& input)
{
    // a = relu(fc1(input))
    VectorXf a{ (actor.fc1.weight * input + actor.fc1.bias).array().max(0) };
    // a = relu(fc2(a))
    a = (actor.fc2.weight * a + actor.fc2.bias).array().max(0);
    // mu = mu(a)
    double mu{ (actor.mu.weight * a + actor.mu.bias).value() };
    // log_std = tanh(sigma(a))
    double sigma{ ((actor.sigma.weight * a + actor.sigma.bias).array().tanh()).value() };
    // maybe making a new variable here might make it consume more memory
    sigma = exp(LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (sigma + 1));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(mu, sigma);

    float action{ static_cast<float>(tanh(distribution(gen))) };
    return action;
}
#else
// upon testing on 2/4/2024, returning a float is faster than returning an Eigen vector
float feedForward(const Agent& actor, const VectorXf& input)
{
    // a = relu(fc1(input))
    VectorXf a{ (actor.fc1.weight * input + actor.fc1.bias).array().max(0) };
#ifdef LAYER_NORMALIZATION
    // a = layerNorm1(a)
    float mu{ a.array().mean() };
    float var{ ((a.array() - mu) * (a.array() - mu)).mean() };
    float std{ numext::sqrt(var + actor.layerNormEpsilon) };
    a = (a.array() - mu) / std;
    a = actor.ln1.weight.array() * a.array() + actor.ln1.bias.array();
#endif
    // a = relu(fc2(a))
    a = (actor.fc2.weight * a + actor.fc2.bias).array().max(0);
#ifdef LAYER_NORMALIZATION
    // a = layerNorm2(a)
    mu = a.array().mean();
    var = ((a.array() - mu) * (a.array() - mu)).mean();
    std = numext::sqrt(var + actor.layerNormEpsilon);
    a = (a.array() - mu) / std;
    a = actor.ln2.weight.array() * a.array() + actor.ln2.bias.array();
#endif
    // mu = tanh(mu(a))
    return (actor.mu.weight * a + actor.mu.bias).array().tanh().value();
}
#endif

void saveMemory(const MatrixXf& observations, const MatrixXf& actions)
{
    using namespace std;
    IOFormat Format(FullPrecision, DontAlignCols, ",", "\n", "", "");
    string observationsFilename{ "observations.csv" };
    string observationsFilepath{ "memory/" + observationsFilename };
    {
        ofstream file(static_cast<string>(observationsFilepath), ios::out);
        if (!file.is_open()) cout << "Error opening observations csv file.\n";
        else
        {
//            for (int i{ 0 }; i < observations.rows(); ++i)
//            {
//                for (int j{ 0 }; j < observations.cols(); ++j)
//                {
//                    file << std::to_string(observations(i, j));
//                    if (j < (observations.cols() - 1)) file << ",";
//                } file << "\n";
//            }
            file << observations.format(Format);
            file.close();
        }
    }
    string actionsFilename{ "actions.csv" };
    string actionsFilepath{ "memory/" + actionsFilename };
    ofstream file( static_cast<string>(actionsFilepath), ios::out);
    if (!file.is_open()) cout << "Error opening observations csv file.\n";
    else
    {
//        for (int i{ 0 }; i < actions.rows(); ++i)
//        {
//            file << std::to_string(actions(i, 0)) << "\n";
//        }
        file << actions.format(Format);
        file.close();
    }

    // zip data into one file
    // zip -r memory/td3_fork_memory.zip memory/observations.csv memory/actions.csv
    int returnCode{ std::system(("cd memory; zip -r " + MEMORY_FILENAME + " " + observationsFilename + " " + actionsFilename).c_str()) };
    if (returnCode != 0) std::perror("Error zipping memory files\n");
    // delete data csv files
    returnCode = std::system(("rm -f " + observationsFilepath + " " + actionsFilepath).c_str());
    if (returnCode != 0) std::perror("Error deleting memory files\n");
}
