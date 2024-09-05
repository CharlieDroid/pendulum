//
// Created by Charles on 8/14/2024.
//

#include "agent.h"


//Agent initActor() {
//    Agent actor;
//    // Initialize the weights and biases with random values
//    actor.fc1.weight.setRandom();
//    actor.fc1.bias.setRandom();
//
//    actor.fc2.weight.setRandom();
//    actor.fc2.bias.setRandom();
//
//    actor.mu.weight.setRandom();
//    actor.mu.bias.setRandom();
//
//    // Optionally, initialize layer normalization parameters with random values
//    actor.ln1.weight.setRandom();
//    actor.ln1.bias.setRandom();
//
//    actor.ln2.weight.setRandom();
//    actor.ln2.bias.setRandom();
//
//    return actor;
//}
//
//float feedForward(const Agent& actor, const InputVector& input)
//{
//    // a = relu(fc1(input))
//    VectorXf a{ (actor.fc1.weight * input + actor.fc1.bias).array().max(0) };
//
//    // a = layerNorm1(a)
//    float mu{ a.array().mean() };
//    float var{ ((a.array() - mu) * (a.array() - mu)).mean() };
//    float std{ numext::sqrt(var + LN_EPSILON) };
//    a = (a.array() - mu) / std;
//    a = actor.ln1.weight.array() * a.array() + actor.ln1.bias.array();
//
//    // a = relu(fc2(a))
//    a = (actor.fc2.weight * a + actor.fc2.bias).array().max(0);
//
//    // a = layerNorm2(a)
//    mu = a.array().mean();
//    var = ((a.array() - mu) * (a.array() - mu)).mean();
//    std = numext::sqrt(var + LN_EPSILON);
//    a = (a.array() - mu) / std;
//    a = actor.ln2.weight.array() * a.array() + actor.ln2.bias.array();
//
//    // mu = tanh(mu(a))
//    return (actor.mu.weight * a + actor.mu.bias).array().tanh().value();
//}

Agent initActor() {
    Agent actor;
    // Initialize the weights and biases with random values
    actor.fc1_weight.setRandom();
    actor.fc1_bias.setRandom();

    actor.fc2_weight.setRandom();
    actor.fc2_bias.setRandom();

    actor.mu_weight.setRandom();
    actor.mu_bias.setRandom();

    // Optionally, initialize layer normalization parameters with random values
    actor.ln1_weight.setRandom();
    actor.ln1_bias.setRandom();

    actor.ln2_weight.setRandom();
    actor.ln2_bias.setRandom();

    return actor;
}

float feedForward(const Agent& actor, const InputVector& input)
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
