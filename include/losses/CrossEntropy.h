#pragma once

#include <vector>

using namespace std;

class CrossEntropy {
    private:
        static const double CROSS_ENTROPY_EPSILON;

    public:
        double calculateLoss(int, const vector<double>&);    
        vector<double> calculateGradient(int, const vector<double>&) const;
};