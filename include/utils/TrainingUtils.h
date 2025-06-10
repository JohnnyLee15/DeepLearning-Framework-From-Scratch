#pragma once

#include <vector>

class Matrix;

using namespace std;

class TrainingUtils {
    private:
        // Constants
        static const double GRADIENT_THRESHOLD;
        
    public:
        // Methods
        static double getAccuracy(const vector<double>&, const vector<double>&);
        static double clipDerivative(double);
        static vector<double> getPredictions(const Matrix&);
        static double getPrediction(const vector<double> &, size_t, size_t);
};