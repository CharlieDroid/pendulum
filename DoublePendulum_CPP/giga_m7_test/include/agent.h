//
// Created by Charles on 10/19/2024.
//

#ifndef AGENT_H
#define AGENT_H

#include <Arduino.h>
#include <ArduinoEigenDense.h>

using namespace Eigen;

constexpr int INPUT_SIZE{ 4 };
constexpr int OUTPUT_SIZE{ 1 };
// limit is 100x75 if using static arrays for some reason ??
constexpr int LAYER1_SIZE{ 16 };
constexpr int LAYER2_SIZE{ 16 };
constexpr float LN_EPSILON{ 1e-5f };

typedef Matrix<float, LAYER1_SIZE, INPUT_SIZE> FC1_Weight;
typedef Matrix<float, LAYER1_SIZE, 1> FC1_Bias;
typedef Matrix<float, LAYER2_SIZE, LAYER1_SIZE> FC2_Weight;
typedef Matrix<float, LAYER2_SIZE, 1> FC2_Bias;
typedef Matrix<float, OUTPUT_SIZE, LAYER2_SIZE> MU_Weight;
typedef Matrix<float, OUTPUT_SIZE, 1> MU_Bias;

struct Agent
{
    // Rows = First Size, Cols = Second Size
    FC1_Weight fc1_weight;
    FC1_Bias fc1_bias;

    FC2_Weight fc2_weight;
    FC2_Bias fc2_bias;

    MU_Weight mu_weight;
    MU_Bias mu_bias;

    // Layer Normalization, size of vector is same as biases
    FC1_Bias ln1_weight;
    FC1_Bias ln1_bias;

    FC2_Bias ln2_weight;
    FC2_Bias ln2_bias;
};

typedef Matrix<float, INPUT_SIZE, 1> ObservationVector;

float feedForward(const Agent& actor, const ObservationVector& input);

#endif //AGENT_H
