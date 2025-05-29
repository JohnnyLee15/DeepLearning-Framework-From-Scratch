#pragma once

#include <vector>

class Matrix;

using namespace std;

class TrainingUtils {
    private:
        // Constant
        static const double GRADIENT_THRESHOLD;
        
    public:
        static double getAccuracy(const vector<int>&, const vector<int>&);
        static double clipDerivative(double);
        static vector<int> getPredictions(const Matrix&);
        static int getPrediction(const Matrix&, size_t);
};