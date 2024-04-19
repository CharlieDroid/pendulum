//
// Created by Charles on 2/3/2024.
//

#ifndef RPI_CPP_AGENT_H
#define RPI_CPP_AGENT_H

// defined if layer normalization
//#define LAYER_NORMALIZATION
// default is TD3
//#define SAC

#include <Eigen/Dense>

// constants declared in header file
constexpr int INPUT_SIZE{ 4 };
constexpr int OUTPUT_SIZE{ 1 };
constexpr int LAYER1_SIZE{ 16 };
constexpr int LAYER2_SIZE{ 16 };
#ifdef SAC
constexpr float LOG_STD_MAX{ 2.0f };
constexpr float LOG_STD_MIN{ -5.0f };
#endif

using namespace Eigen;

struct Layer  // Fully connected layer
{
    MatrixXf weight;
    VectorXf bias;
};

#ifdef SAC
struct Agent
{
    Layer fc1{ MatrixXf(LAYER1_SIZE, INPUT_SIZE), VectorXf(LAYER1_SIZE) };
    Layer fc2{ MatrixXf(LAYER2_SIZE, LAYER1_SIZE), VectorXf(LAYER2_SIZE) };
    Layer mu{ MatrixXf(OUTPUT_SIZE, LAYER2_SIZE), VectorXf(OUTPUT_SIZE) };
    Layer sigma{ MatrixXf(OUTPUT_SIZE, LAYER2_SIZE), VectorXf(OUTPUT_SIZE) };
};
#else
struct Agent
{
    Layer fc1{ MatrixXf(LAYER1_SIZE, INPUT_SIZE), VectorXf(LAYER1_SIZE) };
    Layer fc2{ MatrixXf(LAYER2_SIZE, LAYER1_SIZE), VectorXf(LAYER2_SIZE) };
    Layer mu{ MatrixXf(OUTPUT_SIZE, LAYER2_SIZE), VectorXf(OUTPUT_SIZE) };
#ifdef LAYER_NORMALIZATION
    Layer ln1{ MatrixXf(LAYER1_SIZE, 1), VectorXf(LAYER1_SIZE) };
    Layer ln2{ MatrixXf(LAYER2_SIZE, 1), VectorXf(LAYER2_SIZE) };
    float layerNormEpsilon{ 1e-5f };
#endif
};
#endif

Agent initActor();
float feedForward(const Agent& actor, const VectorXf& input);
void saveMemory(const MatrixXf& observations, const MatrixXf& actions);

#endif //RPI_CPP_AGENT_H
