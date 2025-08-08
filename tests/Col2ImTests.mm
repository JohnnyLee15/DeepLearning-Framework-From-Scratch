// #include "core/layers/Conv2D.h"
// #include "core/tensor/Matrix.h"
// #include "utils/Im2ColUtils.h"
// #include <cassert>
// #include <cmath>
// #include <random>
// #include <chrono>
// #import <Metal/Metal.h>
// #import <Foundation/Foundation.h>
// #import "core/gpu/GpuEngine.h"
// #include "core/activations/ReLU.h"
// #include "core/gpu/GpuTypes.h"
// #include <iostream>

// void compareTensors(const Tensor& t1, const Tensor& t2, const std::string& name) {
//     auto v1 = t1.getFlat();
//     auto v2 = t2.getFlat();
    
//     if (v1.size() != v2.size()) {
//         fprintf(stderr, "Assertion failed: Size mismatch in tensor '%s'. CPU size: %zu, GPU size: %zu\n",
//                 name.c_str(), v1.size(), v2.size());
//         assert(false); // Halt execution if sizes are different
//     }
//     for (size_t i = 0; i < v1.size(); ++i) {
//         if (fabs(v1[i] - v2[i]) > 1.0f) { // Use a slightly larger tolerance for backprop
//             printf("Mismatch in %s at idx %zu: CPU=%.6f GPU=%.6f\n", name.c_str(), i, v1[i], v2[i]);
//             assert(false);
//         }
//     }
// }

// /* ========================================================= *
//  *  CPU reference col2im (N,H,W,C layout, SAME math as GPU)  *
//  * ========================================================= */
// static void cpuCol2Im(const Tensor& grad,            // im2col matrix
//                       Tensor& dX,                    // output tensor to fill
//                       size_t oH, size_t oW,          // gradRows / gradCols
//                       size_t kH, size_t kW,
//                       size_t stride,
//                       size_t padT, size_t padL)
// {
//     auto& out = dX.getFlat();                 std::fill(out.begin(), out.end(), 0.f);
//     const auto& g  = grad.getFlat();

//     const size_t N = dX.getShape()[0];
//     const size_t H = dX.getShape()[1];
//     const size_t W = dX.getShape()[2];
//     const size_t C = dX.getShape()[3];
//     const size_t rowStride = kH * kW * C;

//     for (size_t n = 0; n < N; ++n) {
//         for (size_t orow = 0; orow < oH; ++orow)
//         for (size_t ocol = 0; ocol < oW; ++ocol) {
//             const size_t rowIdx = ((n * oH + orow) * oW + ocol) * rowStride;

//             for (size_t kr = 0; kr < kH; ++kr)
//             for (size_t kc = 0; kc < kW; ++kc) {
//                 int inY = int(orow * stride + kr) - int(padT);
//                 int inX = int(ocol * stride + kc) - int(padL);
//                 if (inY < 0 || inY >= int(H) || inX < 0 || inX >= int(W)) continue;

//                 for (size_t c = 0; c < C; ++c) {
//                     size_t col = (kr * kW + kc) * C + c;
//                     size_t gIdx = rowIdx + col;
//                     size_t xIdx = ((n * H + inY) * W + inX) * C + c;
//                     out[xIdx] += g[gIdx];
//                 }
//             }
//         }
//     }
// }

// /* ========================================================= *
//  *  GPU vs CPU comparison / timing wrapper for col2Im        *
//  * ========================================================= */
// void testCol2ImCompare(size_t N, size_t H, size_t W,
//                        size_t C, size_t k, size_t stride,
//                        size_t pad)
// {
//     /* ----- shapes ----- */
//     const size_t oH = (H + 2*pad - k) / stride + 1;
//     const size_t oW = (W + 2*pad - k) / stride + 1;
//     Tensor grad({N * oH * oW, k * k * C});          // im2col shape
//     Tensor dXcpu({N, H, W, C});
//     Tensor dXgpu({N, H, W, C});

//     /* ----- randomise grad ----- */
//     std::mt19937 gen(123); std::uniform_real_distribution<float> dist(-1,1);
//     for (auto& v : grad.getFlat()) v = dist(gen);
//     grad.uploadToGpu();                             // GPU copy
//     dXgpu.uploadToGpu();                            // zero-init

//     /* ----- GPU run ----- */
//     auto q  = GpuEngine::getCmdQueue();
//     auto cb = [q commandBuffer];
//     auto t0 = std::chrono::high_resolution_clock::now();
//     Im2ColUtils::col2Im(grad, dXgpu,
//                         oH, oW, k, k,
//                         stride, pad, pad,
//                         (id<MTLCommandBuffer>)cb);
//     [cb commit]; [cb waitUntilCompleted];
//     auto t1 = std::chrono::high_resolution_clock::now();
//     double gpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

//     /* ----- CPU run ----- */
//     auto t2 = std::chrono::high_resolution_clock::now();
//     cpuCol2Im(grad, dXcpu, oH, oW, k, k, stride, pad, pad);
//     auto t3 = std::chrono::high_resolution_clock::now();
//     double cpuMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

//     /* ----- download & compare ----- */
//     dXgpu.downloadFromGpu();
//     compareTensors(dXcpu, dXgpu, "dX");

//     printf("✅ col2Im CPU vs GPU match!\n");
//     printf("    CPU time: %.3f ms | GPU time: %.3f ms\n", cpuMs, gpuMs);
// }

// /* ========================================================= *
//  *  Quick test-suite like your Conv2D harness                *
//  * ========================================================= */
// void testCol2ImSuite() {
//     testCol2ImCompare(32, 64, 64, 16, 3, 1, 1);
//     testCol2ImCompare(32, 64, 64, 64, 3, 1, 1);
//     testCol2ImCompare(10, 224, 224, 4, 3, 1, 1);
//     testCol2ImCompare(8, 16, 16, 32, 16, 1, 1);
//     testCol2ImCompare(4, 128, 128, 64, 8, 1, 1);

//     /* stride-2 / larger pad variants */
//     testCol2ImCompare(8, 30, 50, 4, 3, 2, 1);
//     testCol2ImCompare(16, 32, 32, 60, 5, 2, 2);
// }

// int main() {
//     GpuEngine::init();
//     testCol2ImSuite();

//     return 0;
// }
