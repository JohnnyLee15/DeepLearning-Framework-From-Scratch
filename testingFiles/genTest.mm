// #include "core/layers/Conv2D.h"
// #include "core/tensor/Matrix.h"
// #include "utils/Im2ColUtils.h"
// #include <cassert>
// #include <cmath>
// #include <random>
// #include <chrono>               // ← for timing
// #import <Metal/Metal.h>
// #import <Foundation/Foundation.h>
// #import "core/gpu/GpuEngine.h"
// #include "core/activations/ReLU.h"
// #include "core/gpu/GpuTypes.h"

// void testConv2DForwardCompare() {
//   //–– 1) Setup a small conv layer
//   size_t N      = 32,
//          H      = 256,
//          W      = 256,
//          C_in   = 3,
//          C_out  = 32,
//          kH     = 3,
//          kW     = 3,
//          stride = 1;

//   Conv2D cpuLayer(C_out, kH, kW, stride, "Same", new ReLU());
//   Conv2D gpuLayer(C_out, kH, kW, stride, "same", new ReLU());

//   //–– 2) Build both
//   vector<size_t> inShape = {N,H,W,C_in};
//   cpuLayer.build(inShape);
//   gpuLayer.build(inShape);

//   //–– 3) Random input
//   Tensor x(inShape);
//   std::mt19937 gen(42);
//   std::uniform_real_distribution<float> d(-1,1);
//   for (auto &v : x.getFlat()) v = d(gen);

//   //–– 4) Time CPU forward
//   auto t0 = std::chrono::high_resolution_clock::now();
//   cpuLayer.forward(x);
//   auto t1 = std::chrono::high_resolution_clock::now();
//   auto cpuOut = cpuLayer.getOutput().getFlat();
//   double cpuMs = std::chrono::duration<double,std::milli>(t1 - t0).count();

//   //–– 5) Upload and time GPU forward
//   x.uploadToGpu();
//   id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();
//   id<MTLCommandBuffer> cmd  = [queue commandBuffer];
//   auto t2 = std::chrono::high_resolution_clock::now();
//   gpuLayer.forwardGpu(x, (GpuCommandBuffer)cmd);
//   [cmd commit];
//   [cmd waitUntilCompleted];
//   auto t3 = std::chrono::high_resolution_clock::now();
//   double gpuMs = std::chrono::duration<double,std::milli>(t3 - t2).count();

//   //–– 6) Copy back GPU result
//   auto &gpuPre = const_cast<Tensor&>(gpuLayer.getOutput());
//   gpuPre.downloadFromGpu();
//   auto gpuOut = gpuPre.getFlat();

//   //–– 7) Compare elementwise
//   assert(cpuOut.size() == gpuOut.size());
//   for (size_t i = 0; i < cpuOut.size(); ++i) {
//     float a = cpuOut[i], b = gpuOut[i];
//     if (fabs(a - b) > 1e-4f) {
//       printf("Mismatch at idx %zu: CPU=%.6f GPU=%.6f\n", i, a, b);
//       assert(false);
//     }
//   }

//   //–– 8) Report timings
//   printf("✅ Conv2D forward CPU vs GPU match!\n");
//   printf("    CPU time: %.3f ms\n", cpuMs);
//   printf("    GPU time: %.3f ms (dispatch + wait)\n", gpuMs);
// }

// int main() {
//   GpuEngine::init();
//   testConv2DForwardCompare();
//   return 0;
// }
