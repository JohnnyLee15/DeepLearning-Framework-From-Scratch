// #include "utils/Im2ColUtils.h"
// #include "core/tensor/Tensor.h"
// #include "core/gpu/GpuEngine.h"
// #include <cassert>
// #include <cmath>
// #include <random>
// #include <chrono>
// #import <Metal/Metal.h>
// #import <Foundation/Foundation.h>

// // This function specifically tests the performance of the im2col GPU kernel.
// void testIm2ColGpu() {
//     double gpuMs = 0;
//     double total_gb = 0;

//     // 1. Setup dimensions consistent with the Conv2D test
//     size_t N      = 32,
//           H      = 256,
//           W      = 256,
//           C_in   = 3,
//           kH     = 3,
//           kW     = 3,
//           stride = 1;

//     // 2. Create and populate the input tensor
//     vector<size_t> inShape = {N, H, W, C_in};
//     Tensor x(inShape);
//     std::mt19937 gen(42);
//     std::uniform_real_distribution<float> d(-1, 1);
//     for (auto &v : x.getFlat()) v = d(gen);

//     // 3. Calculate the expected output shape for the im2col operation
//     // For "Same" padding, the output height and width match the input.
//     size_t H_out = H;
//     size_t W_out = W;
//     size_t im2col_rows = N * H_out * W_out;
//     size_t im2col_cols = kH * kW * C_in;
//     vector<size_t> im2colShape = {im2col_rows, im2col_cols};
//     Tensor im2ColInBuf(im2colShape);

//     // Calculate total data transferred for bandwidth calculation
//     // Data written = size of the output im2col tensor.
//     // Data read = for each output pixel, we read a patch. Total reads = size of output tensor.
//     double total_elements = (double)im2col_rows * im2col_cols;
//     double total_bytes_transferred = (total_elements + total_elements) * sizeof(float); // Read + Write
//     total_gb = total_bytes_transferred / 1e9; // Convert bytes to gigabytes

//     // 4. The im2col kernel needs a WindowDims struct
//     WindowDims win;
//     win.outRows = H_out;
//     win.outCols = W_out;

//     printf("Running im2col GPU performance test...\n");

//     // 5. Run the timing loop
//     for (size_t t = 0; t < 10; t++) {
//         // Ensure data and output buffers are on the GPU
//         x.uploadToGpu();
//         im2ColInBuf.uploadToGpu(); 

//         id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();
//         id<MTLCommandBuffer> cmd  = [queue commandBuffer];

//         // --- Time the isolated im2col call ---
//         auto t0 = std::chrono::high_resolution_clock::now();
//         Im2ColUtils::im2Col(x, im2ColInBuf, kH, kW, stride, win, cmd);
//         [cmd commit];
//         [cmd waitUntilCompleted];
//         auto t1 = std::chrono::high_resolution_clock::now();
        
//         gpuMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
//     }

//     // 6. Report the final average time and bandwidth
//     double avgGpuMs = gpuMs / 10.0;
//     double avgGpuSec = avgGpuMs / 1000.0;
//     double bandwidth = total_gb / avgGpuSec;

//     printf("✅ im2col GPU test complete.\n");
//     printf("    Average GPU time: %.3f ms (Bandwidth: %.2f GB/s)\n", avgGpuMs, bandwidth);
// }

// int main() {
//   GpuEngine::init();
//   testIm2ColGpu();
//   return 0;
// }
