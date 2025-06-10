#pragma once
#include "utils/Scalar.h"

class Minmax : public Scalar {
    private:
        // Instance Variables
        vector<double> minVals;
        vector<double> maxVals;

    public:
        // Methods
        void fit(const Matrix&) override;
        void transform(Matrix&) override;
        void reverseTransform(Matrix&) const override;
        void fit(const vector<double>&) override;
        void transform(vector<double>&) override;
        void reverseTransform(vector<double>&) const override;
};