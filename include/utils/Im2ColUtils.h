#pragma once

#include <cstddef>
#include "core/tensor/Tensor.h"

class Im2ColUtils{
   
    public:
        static size_t getGpuFastSize();
        static size_t getTileSize();

         #ifdef __OBJC__
            static void im2Col(
                const Tensor&, Tensor&, size_t, size_t, size_t, 
                const WindowDims&, id<MTLCommandBuffer>
            );

            static void col2Im(
                const Tensor&, Tensor&, size_t, size_t,
                size_t, size_t, size_t, size_t, size_t, 
                id<MTLCommandBuffer>
            );

            static void addBiasApplyReLUIm2Col(Tensor&, Tensor&, const Tensor&, id<MTLCommandBuffer>);
        
        #endif
};