#pragma once

#include <vector>

class Matrix;

using namespace std;

class Activation {
    public:
        // Methods
        virtual Matrix activate(const Matrix&) const = 0;
        virtual Matrix calculateGradient(const Matrix&) const = 0;
        virtual vector<double> initBias(size_t) const = 0;
        virtual ~Activation() = default;
        virtual bool isFused() const;
};