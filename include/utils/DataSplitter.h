#pragma once

#include "core/tensor/Tensor.h"
#include <vector>

using namespace std;

struct Split {
    Tensor xTrain;
    Tensor xVal;
    vector<float> yTrain;
    vector<float> yVal;

    void clear();
};

class DataSplitter {
    private:
        static Split prepareSplit(size_t, size_t, size_t, const Tensor&);
        static float clampRatio(float);

    public:
        static Split stratifiedSplit(const Tensor&, const vector<float>&, float valRatio = 0.10f);
        static Split randomSplit(const Tensor&, const vector<float>&, float valRatio = 0.10f);
};