// #include "core/layers/MaxPooling2D.h"
// #include "core/tensor/Tensor.h"
// #include "utils/ConsoleUtils.h"
// #include "core/gpu/GpuEngine.h"
// #include "core/gpu/GpuTypes.h"

// #include <cassert>
// #include <chrono>
// #include <cmath>
// #include <cstdio>
// #include <iostream>
// #include <random>
// #include <string>
// #include <vector>

// #ifdef __APPLE__
// #import <Foundation/Foundation.h>
// #import <Metal/Metal.h>
// #endif

// // Helper function to compare two tensors
// void compareTensors(const Tensor& t1, const Tensor& t2, const std::string& name) {
//     auto v1 = t1.getFlat();
//     auto v2 = t2.getFlat();
    
//     if (v1.size() != v2.size()) {
//         fprintf(stderr, "Assertion failed: Size mismatch in tensor '%s'. CPU size: %zu, GPU size: %zu\n",
//                 name.c_str(), v1.size(), v2.size());
//         assert(false);
//     }
//     for (size_t i = 0; i < v1.size(); ++i) {
//         if (fabs(v1[i] - v2[i]) > 1e-4f) {
//             printf("Mismatch in %s at idx %zu: CPU=%.6f GPU=%.6f\n", name.c_str(), i, v1[i], v2[i]);
//             assert(false);
//         }
//     }
// }

// // Test function for MaxPooling2D backpropagation
// void testMaxPool2DBackpropCompare(
//     size_t N, size_t H, size_t W, size_t C, 
//     size_t kH, size_t kW, size_t stride, const std::string& padding
// ) {

//     double gpuMs = 0;
//     double cpuMs = 0;

//     // Run multiple iterations for stable timing
//     for (size_t t = 0; t < 10; t++) {
//         // 1. Setup CPU and GPU MaxPooling2D layers
//         vector<size_t> inShape = {N, H, W, C};

//         GpuEngine::disableGpu();
//         MaxPooling2D cpuLayer(kH, kW, stride, padding);
//         cpuLayer.build(inShape);

//         GpuEngine::enableGpu();
//         MaxPooling2D gpuLayer(kH, kW, stride, padding);
//         gpuLayer.build(inShape);

//         // 2. Create random input tensor and upstream gradient tensor
//         Tensor x(inShape);
//         Tensor grad(cpuLayer.getOutput().getShape());

//         std::mt19937 gen(42 + t);
//         std::uniform_real_distribution<float> d(-1.0f, 1.0f);
//         for (auto &v : x.getFlat()) v = d(gen);
//         for (auto &v : grad.getFlat()) v = d(gen);
        
//         x.uploadToGpu();
//         grad.uploadToGpu();

//         // 3. Run forward passes to store the max indices needed for backprop
//         GpuEngine::disableGpu();
//         cpuLayer.forward(x);

//         GpuEngine::enableGpu();
//         #ifdef __APPLE__
//             id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();
//             id<MTLCommandBuffer> fwdCmd = [queue commandBuffer];
//             gpuLayer.forwardGpu(x, (GpuCommandBuffer)fwdCmd);
//             [fwdCmd commit];
//             [fwdCmd waitUntilCompleted];
//         #endif

//         // 4. Time CPU backpropagation
//         GpuEngine::disableGpu();
//         auto t0 = std::chrono::high_resolution_clock::now();
//         cpuLayer.backprop(x, 0.0f, grad, false);
//         auto t1 = std::chrono::high_resolution_clock::now();
//         cpuMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

//         // 5. Time GPU backpropagation
//         GpuEngine::enableGpu();
//         #ifdef __APPLE__
//             id<MTLCommandBuffer> bwdCmd = [queue commandBuffer];
//             auto t2 = std::chrono::high_resolution_clock::now();
//             gpuLayer.backpropGpu(x, 0.0f, grad, false, (GpuCommandBuffer)bwdCmd);
//             [bwdCmd commit];
//             [bwdCmd waitUntilCompleted];
//             auto t3 = std::chrono::high_resolution_clock::now();
//             gpuMs += std::chrono::duration<double, std::milli>(t3 - t2).count();
//         #endif

//         // 6. Download GPU result (the output gradient dX)
//         gpuLayer.getOutputGradient().downloadFromGpu();

//         // 7. Compare the output gradients (dX)
//         compareTensors(cpuLayer.getOutputGradient(), gpuLayer.getOutputGradient(), "dX");
//     }

//     // 8. Report success and timings
//     printf("✅ MaxPool2D backprop CPU vs GPU match! (N=%zu, H=%zu, W=%zu, C=%zu, k=%zux%zu, s=%zu, pad=%s)\n",
//            N, H, W, C, kH, kW, stride, padding.c_str());
//     printf("    CPU time: %.3f ms\n", cpuMs / 10.0);
//     printf("    GPU time: %.3f ms (dispatch + wait)\n", gpuMs / 10.0);
// }

// void test() {
//     printf("--- Testing MaxPooling2D Backpropagation ---\n");
    
//     // Standard cases
//     testMaxPool2DBackpropCompare(8,  32,  32,  16, 2, 2, 2, "none");
//     testMaxPool2DBackpropCompare(16, 64,  64,  32, 2, 2, 2, "none");
    
//     // Non-square kernels and inputs
//     testMaxPool2DBackpropCompare(4,  30,  20,  4,  3, 2, 1, "none");
//     testMaxPool2DBackpropCompare(2,  15,  25,  8,  2, 4, 1, "none");

//     // "Same" padding tests
//     testMaxPool2DBackpropCompare(8,  32,  32,  16, 2, 2, 2, "same");
//     testMaxPool2DBackpropCompare(4,  27,  27,  8,  3, 3, 2, "same"); 
//     testMaxPool2DBackpropCompare(2,  13,  17,  4,  3, 3, 3, "none");

//     // Overlapping pooling (stride < kernel size)
//     testMaxPool2DBackpropCompare(4,  28,  28,  8,  3, 3, 1, "none");
//     testMaxPool2DBackpropCompare(4,  28,  28,  8,  3, 3, 2, "none");

//     // Strides larger than kernel size
//     testMaxPool2DBackpropCompare(4,  16,  16,  8,  2, 2, 3, "none");
//     testMaxPool2DBackpropCompare(4,  16,  16,  8,  2, 2, 3, "same");

//     // Tiny edge cases
//     testMaxPool2DBackpropCompare(1,   1,   1,   1,  1, 1, 1, "none");
//     testMaxPool2DBackpropCompare(1,   1,   1,   1,  1, 1, 1, "same");
//     testMaxPool2DBackpropCompare(2,   2,   2,   2,  2, 2, 2, "none");
//     testMaxPool2DBackpropCompare(2,   2,   2,   2,  2, 2, 2, "same");

//     // Larger, more realistic test cases
//     testMaxPool2DBackpropCompare(32, 56,  56,  64, 3, 3, 2, "none");
//     testMaxPool2DBackpropCompare(16, 128, 128, 64, 2, 2, 2, "none");
//     testMaxPool2DBackpropCompare(4, 224, 224, 3,  3, 3, 2, "none");

//     // -------------------------------------------------------------------
//     // 1) Row‐vector pooling (H=1, W=10), small kernel with stride 2
//     testMaxPool2DBackpropCompare(2, 1, 10, 1, 1, 3, 2, "none");
//     testMaxPool2DBackpropCompare(2, 1, 10, 1, 1, 3, 2, "same");

//     // 2) Column‐vector pooling (H=10, W=1), small kernel with stride 2
//     testMaxPool2DBackpropCompare(2, 10, 1, 1, 3, 1, 2, "none");
//     testMaxPool2DBackpropCompare(2, 10, 1, 1, 3, 1, 2, "same");

//     // 3) Kernel == full spatial extent (5×5 over 5×5), both paddings
//     testMaxPool2DBackpropCompare(3, 5,  5, 3, 5, 5, 1, "none");
//     testMaxPool2DBackpropCompare(3, 5,  5, 3, 5, 5, 1, "same");

//     // 4) Stride larger than input dims (s=5 on 4×4)
//     testMaxPool2DBackpropCompare(1, 4,  4, 2, 3, 3, 5, "none");
//     testMaxPool2DBackpropCompare(1, 4,  4, 2, 3, 3, 5, "same");

//     // 5) High channel count on small spatial (C=16 on 8×8)
//     testMaxPool2DBackpropCompare(1, 8,  8, 16, 2, 2, 2, "none");

//     // 6) Prime‐sized spatial with overlapping large kernel (7×7, k=4×4, s=3)
//     testMaxPool2DBackpropCompare(1, 7,  7, 3, 4, 4, 3, "same");

//     // 7) Kernel larger than one dimension (5×3 over 3×10), same padding
//     testMaxPool2DBackpropCompare(1, 3, 10, 1, 5, 3, 1, "same");
//     testMaxPool2DBackpropCompare(1, 10, 3, 1, 3, 5, 1, "same");

//     // 8) Rectangular overlapping (H=3, W=8, k=3×5, s=1, same)
//     testMaxPool2DBackpropCompare(1, 3,  8, 1, 3, 5, 1, "same");
//     testMaxPool2DBackpropCompare(2,   5,   5,   3,   7,   2, 1, "same");
//     testMaxPool2DBackpropCompare(8,  30,  50,   4,   10,   2, 2, "same");
//     testMaxPool2DBackpropCompare(8,  30,  50,   4,   11,   2, 2, "none");
//     testMaxPool2DBackpropCompare(16, 32,  32,   3,  16,   2, 2, "same");
//     testMaxPool2DBackpropCompare(16, 32,  32,   3,  16,   2, 2, "none");
//     testMaxPool2DBackpropCompare(32, 64,  64,   3,  16,   2, 2, "same");
//     testMaxPool2DBackpropCompare(32, 64,  64,   3,  16,   2, 2, "none");
//     testMaxPool2DBackpropCompare(32, 64,  64,  16,  32,   2, 4, "none");
//     testMaxPool2DBackpropCompare(32, 64,  64,  64,  64,   2, 4, "none");
//     testMaxPool2DBackpropCompare(10,224, 224,   3,  64,   3, 3, "same");
//     testMaxPool2DBackpropCompare(10,224, 224,   3,  64,   6, 3, "none");
//     testMaxPool2DBackpropCompare(8,  16,  16,  32,  32,   5, 4, "none");
//     testMaxPool2DBackpropCompare(8,  16,  16,  64,  64,   4, 4, "none");
//     testMaxPool2DBackpropCompare(4, 128, 128,  64, 128,   3, 4, "same");
//     testMaxPool2DBackpropCompare(2,   7,   7,   3,   5,   2, 2, "none");
//     testMaxPool2DBackpropCompare(2,   9,   9,   3,   5,   1, 4, "same");


//     printf("--------------------------------------------\n");
// }

// int main() {
//     #ifdef __APPLE__
//         GpuEngine::init();
//     #else
//         printf("GPU tests are only available on Apple platforms.\n");
//         return 0;
//     #endif

//     test();

//     return 0;
// }
