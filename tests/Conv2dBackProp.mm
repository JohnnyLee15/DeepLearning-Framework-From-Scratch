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
    
//     if (v1.size() != v2.size()) {
//         fprintf(stderr, "Assertion failed: Size mismatch in tensor '%s'. CPU size: %zu, GPU size: %zu\n",
//                 name.c_str(), v1.size(), v2.size());
//         assert(false); // Halt execution if sizes are different
//     }
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
// Tensor convertGpuDwToTensorFormat(const Tensor& gpuDw, const vector<size_t> &shape) {
//     if (gpuDw.getRank() == 4) {
//         return gpuDw;
//     }

//     size_t numKernals = shape[0];
//     size_t kRows = shape[1];
//     size_t kCols = shape[2];
//     size_t inDepth = shape[3];

//     // Create a new tensor with the 2D matrix shape
//     Tensor gpuDwTensor({numKernals, kRows, kCols, inDepth});
    
//     const auto& gpuDwFlat = gpuDw.getFlat();
//     auto& gpuDwTensorFlat = gpuDwTensor.getFlat();
//     for (size_t o = 0; o < numKernals; o++) {
//         for (size_t i = 0; i < kRows; i++) {
//             for (size_t j = 0; j < kCols; j++) {
//                 for (size_t d  = 0; d < inDepth; d++) {
//                     size_t kIdx = ((o * kRows + i) * kCols + j) * inDepth + d;
//                     size_t row = (i * kCols + j) * inDepth + d;
//                     size_t col = o;
//                     gpuDwTensorFlat[kIdx] = gpuDwFlat[row * numKernals + col];
//                 }
//             }
//         }
//     }
//     return gpuDwTensor;
// }


// void testConv2DBackpropCompare(
//     size_t N, size_t H, size_t W, size_t C_in, size_t C_out, size_t stride
// ) {

//     double gpuMs = 0;
//     double cpuMs = 0;

//     for (size_t t = 0; t < 10; t++) {
//         //–– 1) Setup a conv layer (same as forward test)
//         size_t kH     = 1, kW     = 2;
//         vector<size_t> inShape = {N, H, W, C_in};

//         GpuEngine::disableGpu();
//         Conv2D cpuLayer(C_out, kH, kW, stride, "None", new ReLU(), 0.0001f);
//         cpuLayer.build(inShape);

//         GpuEngine::enableGpu();
//         Conv2D gpuLayer(C_out, kH, kW, stride, "None", new ReLU(), 0.0001f);
//         gpuLayer.build(inShape);

//         //–– 2) Random input and upstream gradient
//         Tensor x(inShape);
//         Tensor grad(cpuLayer.getOutput().getShape());

//         std::mt19937 gen(42);
//         std::uniform_real_distribution<float> d(-1, 1);
//         for (auto &v : x.getFlat()) v = d(gen);
//         for (auto &v : grad.getFlat()) v = d(gen);
//         x.uploadToGpu();
//         grad.uploadToGpu();

//         //–– 3) Run forward passes to set internal state
//         GpuEngine::disableGpu();
//         cpuLayer.forward(x);

//         GpuEngine::enableGpu();
//         id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();
//         id<MTLCommandBuffer> fwdCmd = [queue commandBuffer];
//         gpuLayer.forwardGpu(x, (GpuCommandBuffer)fwdCmd);
//         [fwdCmd commit];
//         [fwdCmd waitUntilCompleted];

//         //–– 4) Time CPU backprop
//         GpuEngine::disableGpu();
//         auto t0 = std::chrono::high_resolution_clock::now();
//         cpuLayer.backprop(x, 0.01f, grad, false);
//         auto t1 = std::chrono::high_resolution_clock::now();
//         cpuMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

//         //–– 5) Upload grad and time GPU backprop
//         GpuEngine::enableGpu();
//         id<MTLCommandBuffer> bwdCmd = [queue commandBuffer];
//         auto t2 = std::chrono::high_resolution_clock::now();
//         gpuLayer.backpropGpu(x, 0.01f, grad, false, (GpuCommandBuffer)bwdCmd);
//         [bwdCmd commit];
//         [bwdCmd waitUntilCompleted];
//         auto t3 = std::chrono::high_resolution_clock::now();
//         gpuMs += std::chrono::duration<double, std::milli>(t3 - t2).count();

//         //–– 6) Download GPU results
//         const_cast<Tensor&>(gpuLayer.getWeights()).downloadFromGpu();
//         const_cast<Tensor&>(gpuLayer.getBiases()).downloadFromGpu();
//         const_cast<Tensor&>(gpuLayer.getDeltaInputs()).downloadFromGpu();

//         //–– 7) Compare gradients
//         // Convert the CPU dW to the GPU's matrix format before comparing.
//         Tensor gpuDwAsTensor = convertGpuDwToTensorFormat(gpuLayer.getWeights(), cpuLayer.getWeights().getShape());
//         compareTensors(cpuLayer.getWeights(), gpuDwAsTensor, "dW");
        
//         compareTensors(cpuLayer.getBiases(), gpuLayer.getBiases(), "biases");
//         compareTensors(cpuLayer.getDeltaInputs(), gpuLayer.getDeltaInputs(), "dX");
//     }

//     //–– 8) Report timings
//     printf("✅ Conv2D backprop CPU vs GPU match!\n");
//     printf("    CPU time: %.3f ms\n", cpuMs / 10.0);
//     printf("    GPU time: %.3f ms (dispatch + wait)\n", gpuMs / 10.0);
// }

// void test() {
//     testConv2DBackpropCompare(1,   1,   2,   1,   1,   2);
//     testConv2DBackpropCompare(1,   1,   2,   1,   1,   2);
//     testConv2DBackpropCompare(1,   2,   2,   1,   1,   2);
//     testConv2DBackpropCompare(1,   2,   2,   1,   1,   2);
//     testConv2DBackpropCompare(2,   5,   5,   1,   1,   2);
//     testConv2DBackpropCompare(2,   5,   5,   1,   1,   2);
//     testConv2DBackpropCompare(2,   5,   5,   3,   3,   2);
//     testConv2DBackpropCompare(4,   8,  16,   3,   8,   2);
//     testConv2DBackpropCompare(4,   8,  16,   3,   8,   2);
//     testConv2DBackpropCompare(8,  30,  50,   4,   4,   2);
//     testConv2DBackpropCompare(8,  30,  50,   4,   4,   2);
//     testConv2DBackpropCompare(16, 32,  32,   3,  16,   2);
//     testConv2DBackpropCompare(16, 32,  32,   3,  16,   2);
//     testConv2DBackpropCompare(32, 64,  64,   3,  16,   2);
//     testConv2DBackpropCompare(32, 64,  64,   3,  16,   2);
//     testConv2DBackpropCompare(32, 64,  64,  16,  32,   2);
//     testConv2DBackpropCompare(32, 64,  64,  64,  64,   2);
//     testConv2DBackpropCompare(10,224, 224,   3,  64,   2);
//     testConv2DBackpropCompare(10,224, 224,   3,  64,   2);
//     testConv2DBackpropCompare(8,  16,  16,  32,  32,   2);
//     testConv2DBackpropCompare(8,  16,  16,  64,  64,   2);
//     testConv2DBackpropCompare(4, 128, 128,  64, 128,   2);
//     testConv2DBackpropCompare(2,   7,   7,   3,   5,   3);
//     testConv2DBackpropCompare(2,   9,   9,   3,   5,   4);
// }

// int main() {
//     GpuEngine::init();
//     test();

//     return 0;
// }
