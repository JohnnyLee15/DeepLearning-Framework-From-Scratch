#pragma once

#include "core/tensor/Tensor.h"

struct Split {
    Tensor xTrain;
    Tensor xVal;
    Tensor yTrain;
    Tensor yVal;
};

class DataSplitter {
    public:
        static Split stratifiedSplit(const Tensor&, const Tensor&, float valRatio = 0.10f);
        static Split randomSplit(const Tensor&, const Tensor&, float valRatio = 0.10f);
};