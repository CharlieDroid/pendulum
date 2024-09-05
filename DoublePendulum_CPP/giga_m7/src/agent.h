//
// Created by Charles on 8/14/2024.
//

#ifndef GIGA_M7_AGENT_H
#define GIGA_M7_AGENT_H

#include <Arduino.h>
#include <ArduinoEigenDense.h>
#include <array>

constexpr int INPUT_SIZE{ 10 };
constexpr int OUTPUT_SIZE{ 1 };
constexpr int LAYER1_SIZE{ 100 };
constexpr int LAYER2_SIZE{ 75 };
constexpr float LN_EPSILON{ 1e-5f };

using namespace Eigen;

//struct Layer  // Fully connected layer
//{
//    MatrixXf weight;
//    VectorXf bias;
//};
//
//struct Agent
//{
//    Layer fc1{ MatrixXf(LAYER1_SIZE, INPUT_SIZE), VectorXf(LAYER1_SIZE) };
//    Layer fc2{ MatrixXf(LAYER2_SIZE, LAYER1_SIZE), VectorXf(LAYER2_SIZE) };
//    Layer mu{ MatrixXf(OUTPUT_SIZE, LAYER2_SIZE), VectorXf(OUTPUT_SIZE) };
//    Layer ln1{ MatrixXf(LAYER1_SIZE, 1), VectorXf(LAYER1_SIZE) };
//    Layer ln2{ MatrixXf(LAYER2_SIZE, 1), VectorXf(LAYER2_SIZE) };
//};

struct Agent
{
    Matrix<float, LAYER1_SIZE, INPUT_SIZE> fc1_weight;
    Matrix<float, LAYER1_SIZE, 1> fc1_bias;
    Matrix<float, LAYER2_SIZE, LAYER1_SIZE> fc2_weight;
    Matrix<float, LAYER2_SIZE, 1> fc2_bias;
    Matrix<float, OUTPUT_SIZE, LAYER2_SIZE> mu_weight;
    Matrix<float, OUTPUT_SIZE, 1> mu_bias;
    Matrix<float, LAYER1_SIZE, 1> ln1_weight;
    Matrix<float, LAYER1_SIZE, 1> ln1_bias;
    Matrix<float, LAYER2_SIZE, 1> ln2_weight;
    Matrix<float, LAYER2_SIZE, 1> ln2_bias;
};

typedef Matrix<float, INPUT_SIZE, 1> InputVector;

Agent initActor();
float feedForward(const Agent& actor, const InputVector& input);

#endif //GIGA_M7_AGENT_H
