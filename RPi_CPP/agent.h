//
// Created by Charles on 2/3/2024.
//

#ifndef RPI_CPP_AGENT_H
#define RPI_CPP_AGENT_H

#include <Eigen/Dense>

// constants declared in header file
constexpr int INPUT_SIZE{ 4 };
constexpr int OUTPUT_SIZE{ 1 };
constexpr int LAYER1_SIZE{ 16 };
constexpr int LAYER2_SIZE{ 16 };

using namespace Eigen;

struct Layer  // Fully connected layer
{
    MatrixXf weight;
    VectorXf bias;
};

struct Agent
{
    Layer fc1{ MatrixXf(LAYER1_SIZE, INPUT_SIZE), VectorXf(LAYER1_SIZE) };
    Layer fc2{ MatrixXf(LAYER2_SIZE, LAYER1_SIZE), VectorXf(LAYER2_SIZE) };
    Layer mu{ MatrixXf(OUTPUT_SIZE, LAYER2_SIZE), VectorXf(OUTPUT_SIZE) };
};

Agent initActor();
float feedForward(const Agent& actor, const VectorXf& input);
void saveMemory(const MatrixXf& observations, const MatrixXf& actions);

#endif //RPI_CPP_AGENT_H
