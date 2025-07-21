// testConv2d_batch.mm  – multi-batch sanity test
#import <Metal/Metal.h>
#include "core/gpu/GpuEngine.h"
#include "core/tensor/Tensor.h"

int main() {
    @autoreleasepool {
        /* ---- Metal init ----------------------------------------------- */
        GpuEngine::init();
        id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();
        id<MTLCommandBuffer> cmd  = [queue commandBuffer];

        /* ---- tensors --------------------------------------------------- */
        // Input  N,H,W,C = 3×3×3×1
        //   sample 0 = 1 ..  9   (sum =  45)
        //   sample 1 = 10 .. 18  (sum = 126)
        //   sample 2 = 19 .. 27  (sum = 207)
        Tensor X({3,3,3,1});
        std::vector<float> in(27);
        for (int i = 0; i < 27; ++i) in[i] = static_cast<float>(i + 1);
        X.getFlat() = std::move(in);

        // Kernel  K,R,S,C = 1×3×3×1, all ones
        Tensor W({1,3,3,1});
        W.getFlat() = std::vector<float>(9, 1.0f);

        // Bias K = 1
        Tensor B({1});
        B.getFlat() = {0};

        // Output  N,H',W',K = 3×1×1×1
        Tensor Y({3,1,1,1});
        Y.getFlat() = std::vector<float>(3, 0.0f);

        /* ---- upload & run --------------------------------------------- */
        X.uploadToGpu();  W.uploadToGpu();  B.uploadToGpu();  Y.uploadToGpu();
        X.conv2dForwardGpu(W, /*stride=*/1, Y, B, cmd);
        [cmd commit];  [cmd waitUntilCompleted];

        /* ---- download & print ----------------------------------------- */
        Y.downloadFromGpu();
        auto &out = Y.getFlat();
        printf("Output (NHWC 3×1×1×1):\n");
        for (int n = 0; n < 3; ++n) {
            printf("sample %d → [%.0f]\n", n, out[n]);
        }

        // Expected:
        // sample 0 → [45]
        // sample 1 → [126]
        // sample 2 → [207]
    }
    return 0;
}
