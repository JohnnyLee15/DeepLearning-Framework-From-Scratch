#pragma once
#include "utils/Scalar.h"

class Minmax : public Scalar {
    private:
        // Instance Variables
        vector<double> minVals;
        vector<double> maxVals;

        // Methods
        void checkRank(const Tensor&) const;

    public:
        // Methods
        void fit(const Tensor&) override;
        void transform(Tensor&) override;
        void reverseTransform(Tensor&) const override;
        void fit(const vector<double>&) override;
        void transform(vector<double>&) override;
        void reverseTransform(vector<double>&) const override;
        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
        uint32_t getEncoding() const override;
};