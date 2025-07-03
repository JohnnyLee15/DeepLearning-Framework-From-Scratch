#pragma once
#include "utils/Scalar.h"

class Minmax : public Scalar {
    private:
        // Instance Variables
        vector<double> minVals;
        vector<double> maxVals;

        // Methods
        void checkRank(const Tensor&) const;
        void checkDims(size_t) const;

    public:
        // Methods
        void fit(const Tensor&) override;
        Tensor transform(const Tensor&) const override;
        Tensor reverseTransform(const Tensor&) const override;

        void fit(const vector<double>&) override;
        vector<double> transform(const vector<double>&) const override;
        vector<double> reverseTransform(const vector<double>&) const override;

        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;

        uint32_t getEncoding() const override;

        void reset() override;
};