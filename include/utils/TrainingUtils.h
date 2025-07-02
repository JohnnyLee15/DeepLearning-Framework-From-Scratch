#pragma once

#include <vector>

class Tensor;

using namespace std;

class TrainingUtils {
    private:
        // Constants
        static const double GRADIENT_THRESHOLD;
        
    public:
        // Methods
        static double getAccuracy(const vector<double>&, const vector<double>&);
        static double clipDerivative(double);
        static vector<double> getPredictions(const Tensor&);
        static double getPrediction(const vector<double> &, size_t, size_t);
        static double getRMSE(const Tensor&, const vector<double>&);
};