//
// Created by Charles on 10/19/2024.
//
#include "agent.h"


float feedForward(const Agent& actor, const ObservationVector& input)
{
    // a = relu(fc1(input))
    VectorXf a{ (actor.fc1_weight * input + actor.fc1_bias).array().max(0) };

    // a = layerNorm1(a)
    float mu{ a.array().mean() };
    float var{ ((a.array() - mu) * (a.array() - mu)).mean() };
    float std{ numext::sqrt(var + LN_EPSILON) };
    a = (a.array() - mu) / std;
    a = actor.ln1_weight.array() * a.array() + actor.ln1_bias.array();

    // a = relu(fc2(a))
    a = (actor.fc2_weight * a + actor.fc2_bias).array().max(0);

    // a = layerNorm2(a)
    mu = a.array().mean();
    var = ((a.array() - mu) * (a.array() - mu)).mean();
    std = numext::sqrt(var + LN_EPSILON);
    a = (a.array() - mu) / std;
    a = actor.ln2_weight.array() * a.array() + actor.ln2_bias.array();

    // mu = tanh(mu(a))
    return (actor.mu_weight * a + actor.mu_bias).array().tanh().value();
}

