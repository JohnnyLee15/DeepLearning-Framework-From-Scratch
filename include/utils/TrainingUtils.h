#pragma once

#include <vector>

class Tensor;

using namespace std;

class TrainingUtils {
    private:
        // Constants
        static const float GRADIENT_THRESHOLD;
        
    public:
        // Methods
        static float getAccuracy(const vector<float>&, const vector<float>&);
        static float clipDerivative(float);
        static vector<float> getPredictions(const Tensor&);
        static float getPrediction(const vector<float> &, size_t, size_t);
        static float getRMSE(const Tensor&, const vector<float>&);
};