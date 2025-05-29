#pragma once

#include <vector>
#include "utils/Matrix.h"

using namespace std;

class CrossEntropy {
    private:
        static const double CROSS_ENTROPY_EPSILON;
        double calculateDerivative(double, size_t, size_t) const;

    public:
        double calculateLoss(const vector<int>&, const Matrix&) const;    
        Matrix calculateGradient(const vector<int>&, const Matrix&) const;
};