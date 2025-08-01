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

// // Helper function to compare two tensors
// void compareTensors(const Tensor& t1, const Tensor& t2, const std::string& name) {
//     auto v1 = t1.getFlat();
//     auto v2 = t2.getFlat();
//     assert(v1.size() == v2.size());
//     for (size_t i = 0; i < v1.size(); ++i) {
//         if (fabs(v1[i] - v2[i]) > 1e-3f) { // Use a slightly larger tolerance for backprop
//             printf("Mismatch in %s at idx %zu: CPU=%.6f GPU=%.6f\n", name.c_str(), i, v1[i], v2[i]);
//             assert(false);
//         }
//     }
// }

// // *** NEW HELPER FUNCTION ***
// // This function converts the CPU's 4D dW tensor into the 2D matrix format
// // used by the GPU, mirroring the logic from your initFlatKernals function.
// Tensor convertCpuDwToGpuMatrixFormat(const Tensor& cpuDw) {
//     const auto& shape = cpuDw.getShape();
//     size_t numKernals = shape[0];
//     size_t kRows = shape[1];
//     size_t kCols = shape[2];
//     size_t inDepth = shape[3];

//     // Create a new tensor with the 2D matrix shape
//     Tensor gpuDwMatrix({kRows * kCols * inDepth, numKernals});
    
//     const auto& cpuDwFlat = cpuDw.getFlat();
//     auto& gpuDwMatrixFlat = gpuDwMatrix.getFlat();
//     for (size_t o = 0; o < numKernals; o++) {
//         for (size_t i = 0; i < kRows; i++) {
//             for (size_t j = 0; j < kCols; j++) {
//                 for (size_t d  = 0; d < inDepth; d++) {
//                     size_t kIdx = ((o * kRows + i) * kCols + j) * inDepth + d;
//                     size_t row = (i * kCols + j) * inDepth + d;
//                     size_t col = o;
//                     gpuDwMatrixFlat[row * numKernals + col] = cpuDwFlat[kIdx];
//                 }
//             }
//         }
//     }
//     return gpuDwMatrix;
// }


// void testConv2DBackpropCompare() {

//     double gpuMs = 0;
//     double cpuMs = 0;

//     for (size_t t = 0; t < 10; t++) {
//         //–– 1) Setup a conv layer (same as forward test)
//         size_t N      = 8,
//               H      = 32,
//               W      = 32,
//               C_in   = 3,
//               C_out  = 8,
//               kH     = 3,
//               kW     = 3,
//               stride = 1;

//         Conv2D cpuLayer(C_out, kH, kW, stride, "Same", new ReLU());
//         Conv2D gpuLayer(C_out, kH, kW, stride, "same", new ReLU());

//         vector<size_t> inShape = {N, H, W, C_in};
//         cpuLayer.build(inShape);
//         gpuLayer.build(inShape);

//         //–– 2) Random input and upstream gradient
//         Tensor x(inShape);
//         Tensor grad(cpuLayer.getOutput().getShape());

//         std::mt19937 gen(42);
//         std::uniform_real_distribution<float> d(-1, 1);
//         for (auto &v : x.getFlat()) v = d(gen);
//         for (auto &v : grad.getFlat()) v = d(gen);

//         //–– 3) Run forward passes to set internal state
//         cpuLayer.forward(x);
//         x.uploadToGpu();
//         grad.uploadToGpu();
//         id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();
//         id<MTLCommandBuffer> fwdCmd = [queue commandBuffer];
//         gpuLayer.forwardGpu(x, (GpuCommandBuffer)fwdCmd);
//         [fwdCmd commit];
//         [fwdCmd waitUntilCompleted];

//         //–– 4) Time CPU backprop
//         auto t0 = std::chrono::high_resolution_clock::now();
//         cpuLayer.backprop(x, 0.01f, grad, false);
//         auto t1 = std::chrono::high_resolution_clock::now();
//         cpuMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

//         //–– 5) Upload grad and time GPU backprop
//         id<MTLCommandBuffer> bwdCmd = [queue commandBuffer];
//         auto t2 = std::chrono::high_resolution_clock::now();
//         gpuLayer.backpropGpu(x, 0.01f, grad, false, (GpuCommandBuffer)bwdCmd);
//         [bwdCmd commit];
//         [bwdCmd waitUntilCompleted];
//         auto t3 = std::chrono::high_resolution_clock::now();
//         gpuMs += std::chrono::duration<double, std::milli>(t3 - t2).count();

//         //–– 6) Download GPU results
//         const_cast<Tensor&>(gpuLayer.getDeltaWeightsIm2Col()).downloadFromGpu();
//         const_cast<Tensor&>(gpuLayer.getDeltaBiases()).downloadFromGpu();
//         const_cast<Tensor&>(gpuLayer.getDeltaInputs()).downloadFromGpu();

//         //–– 7) Compare gradients
//         // Convert the CPU dW to the GPU's matrix format before comparing.
//         Tensor cpuDwAsMatrix = convertCpuDwToGpuMatrixFormat(cpuLayer.getDeltaWeights());
//         compareTensors(cpuDwAsMatrix, gpuLayer.getDeltaWeightsIm2Col(), "dW");
        
//         compareTensors(cpuLayer.getDeltaBiases(), gpuLayer.getDeltaBiases(), "dB");
//         compareTensors(cpuLayer.getDeltaInputs(), gpuLayer.getDeltaInputs(), "dX");
//     }

//     //–– 8) Report timings
//     printf("✅ Conv2D backprop CPU vs GPU match!\n");
//     printf("    CPU time: %.3f ms\n", cpuMs / 10.0);
//     printf("    GPU time: %.3f ms (dispatch + wait)\n", gpuMs / 10.0);
// }

// int main() {
//     GpuEngine::init();
//     testConv2DBackpropCompare();
//     return 0;
// }
