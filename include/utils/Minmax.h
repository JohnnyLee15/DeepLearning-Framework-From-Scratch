#pragma once
#include "utils/Scalar.h"

class Minmax : public Scalar {
    private:
        // Instance Variables
        vector<float> minVals;
        vector<float> maxVals;

        // Methods
        void checkRank(const Tensor&) const;
        void checkDims(size_t) const;

    public:
        // Methods
        void fit(const Tensor&) override;
        Tensor transform(const Tensor&) const override;
        Tensor reverseTransform(const Tensor&) const override;

        void fit(const vector<float>&) override;
        vector<float> transform(const vector<float>&) const override;
        vector<float> reverseTransform(const vector<float>&) const override;

        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;

        uint32_t getEncoding() const override;

        void reset() override;
};