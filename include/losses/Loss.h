#pragma once

#include <vector>

class Matrix;

using namespace std;

class Loss {
    public:
        // Methods
        virtual double calculateTotalLoss(const vector<double>&, const Matrix&) const = 0;    
        virtual Matrix calculateGradient(const vector<double>&, const Matrix&) const = 0;
        virtual ~Loss() = default;
        virtual double formatLoss(double) const;
};