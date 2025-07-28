#pragma once

#include <cstddef>
#include "core/tensor/Tensor.h"

class Im2ColUtils{
    #ifdef __OBJC__
    public:
        static void im2Col(
            const Tensor&, Tensor&, size_t, size_t, size_t, 
            const WindowDims&, id<MTLCommandBuffer>
        );

        static void addBiasIm2Col(Tensor&, const Tensor&, id<MTLCommandBuffer>);
    #endif
};