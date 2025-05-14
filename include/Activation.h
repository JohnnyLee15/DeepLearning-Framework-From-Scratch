#pragma once

#include <vector>

using namespace std;

class Activation {
    public:
        virtual vector<double> activate(const vector<double>&) const = 0;
        virtual vector<double> calculateGradient(const vector<double>&) const = 0;
        virtual double initBias() const = 0;
        virtual ~Activation() = default;
};