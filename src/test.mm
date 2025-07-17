// // testMM.mm

// #import <Metal/Metal.h>
// #include "core/GpuEngine.h"
// #include "core/Tensor.h"
// #include "core/MatrixT.h"
// #include "core/Matrix.h"
// #include <vector>
// #include <algorithm>
// #include <iostream>


// using uint = unsigned int;

// int main() {
//     @autoreleasepool {
//         // 1) Initialize the GPU
//         GpuEngine::init();
//         id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();

//         // 2) Create a command buffer
//         id<MTLCommandBuffer> cmd = [queue commandBuffer];

//         // 3) Create and fill three tensors:
//         //    A is 2×3, B is 3×2, C (the result) is 2×2
//         Tensor A({2, 3});
//         Tensor B({3, 2});
//         Tensor C({2, 2});

//         // Fill A = [1,2,3; 4,5,6], B = [7,8; 9,10; 11,12]
//         std::vector<float> valsA = {1,2,3, 4,5,6};
//         std::vector<float> valsB = {7,8, 9,10, 11,12};
//         std::copy(valsA.begin(), valsA.end(), A.getFlat().begin());
//         std::copy(valsB.begin(), valsB.end(), B.getFlat().begin());

//         // Zero-init C
//         std::fill(C.getFlat().begin(), C.getFlat().end(), 0.0f);

//         // 4) Upload to GPU
//         A.uploadToGpu();
//         B.uploadToGpu();
//         C.uploadToGpu();

//         // 5) Run your mm kernel via the Matrix wrapper
//         Matrix MA(A), MB(B);
//         MA.mmGpu(MB, C, cmd);

//         // 6) Commit and wait for completion
//         [cmd commit];
//         [cmd waitUntilCompleted];

//         // 7) Download and print the result
//         C.downloadFromGpu();
//         auto &out = C.getFlat();
//         std::cout << "A × B =\n";
//         for (int i = 0; i < 2; ++i) {
//             for (int j = 0; j < 2; ++j) {
//                 std::cout << out[i*2 + j] << " ";
//             }
//             std::cout << "\n";
//         }
//         // Expected:
//         // 58 64
//         // 139 154
//     }
//     return 0;
// }