#pragma once
#include <vector>
#include <random>

class Layer;
class Activation;

using namespace std;

class Neuron {
    private:
        // Instance Variables
        vector<double> weights;
        double bias;

        // Static Variables
        static random_device rd;
        static mt19937 generator;

        // Methods
        void initWeights(int);

    public:
        Neuron(int, Activation*);
        double calPreActivation(const vector<double>&);
        void updateBias(double);
        int getNumWeights() const;
        double getWeight(int) const;
        void updateWeights(const vector<double>&, double, double);
};