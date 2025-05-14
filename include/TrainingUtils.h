#pragma once

#include <vector>

using namespace std;

class TrainingUtils {
    private:
        // Constant
        static const double GRADIENT_THRESHOLD;
        
    public:
        static double getAccuracy(const vector<double>&, const vector<int>&);
        static double clipDerivative(double);
        static int getPrediction(const vector<double>&);
};